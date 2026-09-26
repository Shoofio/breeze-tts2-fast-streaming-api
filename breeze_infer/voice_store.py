"""On-disk storage for saved voices (data-model.md "Voice file v1", research.md R13).

One JSON file per saved voice, ``<voices_dir>/<id>.voice.json``. Ported from
`api-alignment`'s `breeze_infer/voices.py` -- the atomic write (temporary file, fsync,
`os.replace`, directory fsync: `_rename_with_retry` ~545-554, the fsync helpers
~563-586), the leftover sweep at startup (~503-510), and the write-lock pattern
(~190-194) -- rewritten for this feature's single-file format (that module kept a
directory per voice with the WAV and a `.npy` alongside; R13 stores codes only, one
JSON file).

`sleep`, `clock` (for `created_at`) and `nonce` (for `.del-*`/`.tmp-*` names) are
injected so tests can pass fixed ones; the caller's real ones come from the composition
root (T065), never a module-level global.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from breeze_infer import voice_file
from breeze_infer.voice_file import VoiceFile, VoiceFileError

SUFFIX = ".voice.json"
BREEZE_SUFFIX = ".breeze"
TMP_PREFIX = ".tmp-"
DEL_PREFIX = ".del-"

# A Windows rename fails with PermissionError while another process (antivirus, search
# indexer) briefly holds a handle on the file; a delete waits this many short attempts
# for the handle to go away before giving up (ported from `A:voices.py`).
DELETE_RENAME_ATTEMPTS = 5
DELETE_RENAME_BACKOFF_SECONDS = 0.05


class VoiceExists(Exception):
    """`create()` found the id already on disk, exactly or case-differing."""


@dataclass(frozen=True)
class SkippedVoiceFile:
    """A file `scan()` rejected.

    `name` is the syntactically valid name this file's slot still reserves
    (data-model.md: "A skipped file with a valid name still reserves that name"), taken
    from the *filename* stem regardless of which check actually failed -- even a file
    that fails to parse at all still occupies its name on disk. It's `None` when the
    stem itself isn't a valid name; there's nothing to reserve.
    """

    file: str
    reason: str
    name: str | None


@dataclass(frozen=True)
class ScanResult:
    voices: list[VoiceFile]
    skipped: list[SkippedVoiceFile]
    breeze_count: int


def _rename_with_retry(src: Path, dst: Path, *, sleep: Callable[[float], None]) -> None:
    """`os.replace`, retried briefly on `PermissionError` (ported from `A:voices.py`,
    `_rename_with_retry` ~545-554)."""
    for attempt in range(DELETE_RENAME_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == DELETE_RENAME_ATTEMPTS - 1:
                raise
            sleep(DELETE_RENAME_BACKOFF_SECONDS)


def _write_bytes(path: Path, payload: bytes) -> None:
    """Ported from `A:voices.py` ~563-567: write, then fsync the file itself so its
    content survives a crash before the directory entry pointing at it does."""
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """Ported from `A:voices.py` ~576-586: fsync the directory entry itself (the
    `os.replace` above), best-effort -- some filesystems don't support fsyncing a
    directory at all, and that's not worth failing the whole write over."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _format_utc(moment: datetime) -> str:
    """data-model.md "Voice file v1"'s `created_at` shape: `"2026-09-24T20:15:00Z"`."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class VoiceStore:
    """`scan()`, `create()` and `remove()` over `<voices_dir>/*.voice.json`.

    Keeps its own case-insensitive index of on-disk ids (`_known`), rebuilt by `scan()`
    and kept current by `create()`/`remove()`, so a case-duplicate create is refused
    without re-reading the directory. `_lock` guards that index; `_write_lock` -- taken
    around the file operations themselves (create's atomic write, delete's rename) --
    serializes writes so two of them never interleave, mirroring `A:voices.py`'s own
    split between the two (~190-194): a lookup never waits on a write's fsyncs.
    """

    def __init__(
        self,
        voices_dir: Path,
        *,
        codebook_size: int,
        codec_fingerprint: str,
        events: Any,
        clock: Callable[[], datetime],
        sleep: Callable[[float], None] = time.sleep,
        nonce: Callable[[], str] = lambda: os.urandom(8).hex(),
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self._codebook_size = codebook_size
        self._codec_fingerprint = codec_fingerprint
        self._events = events
        self._clock = clock
        self._sleep = sleep
        self._nonce = nonce
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._known: dict[str, str] = {}  # lowercased id -> id, as actually on disk

    def path_for(self, voice_id: str) -> Path:
        return self.voices_dir / f"{voice_id}{SUFFIX}"

    # ------------------------------------------------------------------ startup

    def scan(self) -> ScanResult:
        """Load every valid `*.voice.json`, in filename order (so "sorts earlier" in
        the case-duplicate rule means exactly this iteration order); skip and report
        the rest; sweep `.tmp-*`/`.del-*` leftovers; count `.breeze` files (BC-29).

        Emits `voices.loaded` and one `voice.skipped` per rejected file.
        """
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        voices: list[VoiceFile] = []
        skipped: list[SkippedVoiceFile] = []
        breeze_count = 0
        known: dict[str, str] = {}

        for entry in sorted(self.voices_dir.iterdir(), key=lambda p: p.name):
            fname = entry.name
            if fname.startswith((TMP_PREFIX, DEL_PREFIX)):
                entry.unlink(missing_ok=True)
                continue
            if fname.endswith(BREEZE_SUFFIX):
                breeze_count += 1
                continue
            if not entry.is_file() or not fname.endswith(SUFFIX):
                continue

            stem = fname[: -len(SUFFIX)]
            name = stem if voice_file.is_valid_name(stem) else None
            if name is not None and name.lower() in known:
                skipped.append(
                    SkippedVoiceFile(
                        file=fname,
                        reason=f"case-duplicate of {known[name.lower()]}{SUFFIX}",
                        name=name,
                    )
                )
                continue
            try:
                raw = entry.read_bytes()
                voice = voice_file.decode(
                    raw,
                    expected_id=stem,
                    codebook_size=self._codebook_size,
                    codec_fingerprint=self._codec_fingerprint,
                )
            except (OSError, VoiceFileError) as exc:
                skipped.append(SkippedVoiceFile(file=fname, reason=str(exc), name=name))
                continue

            voices.append(voice)
            if name is not None:
                known[name.lower()] = name

        with self._lock:
            self._known = known

        self._events.emit(
            "voices.loaded", loaded=len(voices), skipped=len(skipped), breeze_ignored=breeze_count
        )
        for item in skipped:
            self._events.emit("voice.skipped", file=item.file, reason=item.reason)

        return ScanResult(voices=voices, skipped=skipped, breeze_count=breeze_count)

    # ------------------------------------------------------------------- writing

    def create(self, voice: VoiceFile) -> VoiceFile:
        """Atomically write `voice` under `<voices_dir>/<voice.id>.voice.json`.

        Never overwrites: refuses (`VoiceExists`) whenever `voice.id` already names a
        file on disk, including one differing only by case -- the check and the write
        both happen under `_write_lock`, so two concurrent creates of the same id can't
        both pass the check before either writes. `created_at` is stamped here, from the
        injected clock, not taken from `voice` -- it is this store's own record of when
        the write actually happened, not a caller-supplied guess. Returns the record as
        written (with that stamped `created_at`).
        """
        with self._write_lock:
            with self._lock:
                if voice.id.lower() in self._known:
                    raise VoiceExists(voice.id)

            stamped = replace(voice, created_at=_format_utc(self._clock()))
            data = voice_file.encode(
                id=stamped.id,
                ref_text=stamped.ref_text,
                codes=stamped.codes,
                codec_fingerprint=stamped.codec_fingerprint,
                encode_ms=stamped.encode_ms,
                created_at=stamped.created_at,
            )
            final_path = self.path_for(stamped.id)
            tmp_path = self.voices_dir / f"{TMP_PREFIX}{stamped.id}-{self._nonce()}{SUFFIX}"
            try:
                _write_bytes(tmp_path, data)
                os.replace(tmp_path, final_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)
                raise
            _fsync_dir(self.voices_dir)
            with self._lock:
                self._known[stamped.id.lower()] = stamped.id
        return stamped

    # ------------------------------------------------------------------ deleting

    def remove(self, voice_id: str) -> bool:
        """Rename the file to `.del-<id>-<nonce>`, then unlink it; `False` if `voice_id`
        names nothing on disk.

        The index entry is dropped before the rename is attempted, so a concurrent
        lookup either sees the whole file or none of it -- never a name whose file has
        already moved. If the rename itself fails (after retrying), the entry is put
        back and the failure propagates, so the voice stays registered (matching
        `DELETE`'s `500 voice_delete_failed` -- "the voice stays registered").
        """
        key = voice_id.lower()
        with self._lock:
            actual = self._known.pop(key, None)
        if actual is None:
            return False

        path = self.path_for(actual)
        trash = self.voices_dir / f"{DEL_PREFIX}{actual}-{self._nonce()}{SUFFIX}"
        with self._write_lock:
            try:
                _rename_with_retry(path, trash, sleep=self._sleep)
            except FileNotFoundError:
                return True  # already gone from disk; the index entry is all there was
            except OSError:
                with self._lock:
                    self._known[key] = actual
                raise
        trash.unlink(missing_ok=True)
        return True
