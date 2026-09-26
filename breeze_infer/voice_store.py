"""On-disk storage for saved voices (data-model.md "Voice file v1", research.md R13).

One JSON file per saved voice, ``<voices_dir>/<id>.voice.json``. Ported from
`api-alignment`'s `breeze_infer/voices.py` -- the atomic write (temporary file, fsync,
directory fsync: `_rename_with_retry` ~545-554, the fsync helpers ~563-586, though the
commit step is now a no-overwrite hard link rather than `os.replace`), the leftover sweep at startup (~503-510), and the write-lock pattern
(~190-194) -- rewritten for this feature's single-file format (that module kept a
directory per voice with the WAV and a `.npy` alongside; R13 stores codes only, one
JSON file).

`sleep`, `clock` (for `created_at`) and `nonce` (for `.del-*`/`.tmp-*` names) are
injected so tests can pass fixed ones; the caller's real ones come from the composition
root (T065), never a module-level global.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from breeze_infer import voice_file
from breeze_infer.limits import MAX_VOICE_FILE_BYTES
from breeze_infer.voice_file import (
    CREATED_AT_FORMAT,
    CaseInsensitiveNames,
    VoiceFile,
    VoiceFileError,
)

SUFFIX = ".voice.json"
BREEZE_SUFFIX = ".breeze"
TMP_PREFIX = ".tmp-"
DEL_PREFIX = ".del-"

# A Windows rename fails with PermissionError while another process (antivirus, search
# indexer) briefly holds a handle on the file; a delete waits this many short attempts
# for the handle to go away before giving up (ported from `A:voices.py`).
DELETE_RENAME_ATTEMPTS = 5
DELETE_RENAME_BACKOFF_SECONDS = 0.05

# The os.link errors that mean "this filesystem has no hard links" (FAT/exFAT, some
# network shares): EPERM is what Linux returns for vfat, the rest cover other kernels
# and a link across devices. Only these take create's non-atomic fallback; any other
# link failure (EIO, ENOSPC, EMLINK, ...) is a failed write.
_LINKS_UNSUPPORTED = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS}
)


class VoiceExists(Exception):
    """`create()` found the id already taken, exactly or case-differing: by a loaded
    voice, by a skipped file, or by a file already on disk at commit time."""


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


def _write_fsynced(handle: Any, payload: bytes) -> None:
    """Write, then fsync the file itself so its content survives a crash before the
    directory entry pointing at it does (ported from `A:voices.py` ~563-567). Shared by
    the temp-file write and create's exclusive-create fallback."""
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        _write_fsynced(handle, payload)


def _commit_no_overwrite(tmp_path: Path, final_path: Path, data: bytes) -> None:
    """Publish the fsynced `tmp_path` (holding `data`) at `final_path`, raising
    `FileExistsError` if `final_path` already exists -- the on-disk half of create's
    "never overwrites", which the in-memory index alone can't promise: it doesn't see a
    file added after the scan, and on a case-insensitive filesystem it can't see that
    `bob` would land on an existing `Bob.voice.json`.

    `os.link` is atomic and fails with EEXIST when the target exists, including, on a
    case-insensitive filesystem (NTFS, WSL drvfs, macOS), a target differing only by
    case. `os.replace` can't do this: it overwrites by design.

    Only where hard links aren't supported (`_LINKS_UNSUPPORTED`) does this fall back to
    creating `final_path` with O_CREAT|O_EXCL and writing `data` into it. That is just
    as atomic about *existence* -- the kernel refuses an existing file, whatever its
    case on a case-insensitive filesystem -- but not about content: a crash mid-write
    leaves a truncated file, which `scan()` then skips (its name reserved, and removable
    through DELETE) rather than loading. Python has no portable rename-without-replace
    (Linux's renameat2 RENAME_NOREPLACE isn't in the stdlib), so this is the safest
    no-overwrite commit available there. Any other link failure propagates.
    """
    try:
        os.link(tmp_path, final_path)
        return
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno not in _LINKS_UNSUPPORTED:
            raise

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(final_path, flags, 0o644)
    try:
        with open(fd, "wb") as handle:
            _write_fsynced(handle, data)
    except BaseException:
        # O_EXCL means this process created the file, so removing the partial copy can
        # never touch anyone else's.
        final_path.unlink(missing_ok=True)
        raise


def _read_bounded(path: Path) -> bytes:
    """Read at most `MAX_VOICE_FILE_BYTES` (+1, to tell "at the bound" from "over it"),
    so a huge file dropped in the directory can't make startup allocate its whole size
    just to reject it."""
    with open(path, "rb") as handle:
        raw = handle.read(MAX_VOICE_FILE_BYTES + 1)
    if len(raw) > MAX_VOICE_FILE_BYTES:
        raise VoiceFileError(f"larger than {MAX_VOICE_FILE_BYTES} bytes")
    return raw


def _fsync_dir(path: Path) -> None:
    """Ported from `A:voices.py` ~576-586: fsync the directory entries themselves (a
    create's link, a delete's rename), best-effort -- some filesystems don't support fsyncing a
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
    return moment.strftime(CREATED_AT_FORMAT)


class VoiceStore:
    """`scan()`, `create()` and `remove()` over `<voices_dir>/*.voice.json`.

    Two indexes, rebuilt by `scan()` and kept current by `create()`/`remove()`, both
    `CaseInsensitiveNames` of exact name -> path (the registry uses the same structure):
    `_known` for loaded voices, and `_skipped` for skipped entries whose stem is a valid
    name. Together they let `create()` refuse a
    taken name, ignoring case, without re-reading the directory, and let `remove()`
    delete a skipped file as well as a loaded voice's.

    Naming rule (data-model.md "Registry rules"): uniqueness ignores case, but only at
    create; `remove()` matches an id exactly -- the id as `GET /v1/voices` lists it, or
    a skipped file's exact stem -- the same rule `VoiceRegistry.remove` applies.

    `_write_lock` is held across a whole create (check, write, commit) and a whole
    remove (lookup, rename, index update), so no create can slip between a remove's
    lookup and its rename and have its new file renamed away. `_lock` guards the index
    dicts themselves, following `A:voices.py`'s split (~190-194).
    """

    def __init__(
        self,
        voices_dir: Path,
        *,
        codebooks: int,
        codebook_size: int,
        codec_fingerprint: str,
        events: Any,
        clock: Callable[[], datetime],
        sleep: Callable[[float], None] = time.sleep,
        nonce: Callable[[], str] = lambda: os.urandom(8).hex(),
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self._codebooks = codebooks
        self._codebook_size = codebook_size
        self._codec_fingerprint = codec_fingerprint
        self._events = events
        self._clock = clock
        self._sleep = sleep
        self._nonce = nonce
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._known: CaseInsensitiveNames[Path] = CaseInsensitiveNames()
        self._skipped: CaseInsensitiveNames[Path] = CaseInsensitiveNames()

    def path_for(self, voice_id: str) -> Path:
        return self.voices_dir / f"{voice_id}{SUFFIX}"

    def _cleanup_failed(self, entry: Path, op: str, exc: OSError | str) -> None:
        """data-model.md's `voice.cleanup_failed{file, op, error}`: a leftover the
        store couldn't remove or even look at. Never raised: the file is left for the
        next scan's sweep, and whatever the store was doing carries on. `op` names the
        step -- `sweep` (a startup leftover), `stat` (classifying one), `unlink_tmp`
        (create's temp file) or `unlink` (a delete's `.del-*` file)."""
        self._events.emit("voice.cleanup_failed", file=entry.name, op=op, error=str(exc))

    # ------------------------------------------------------------------ startup

    def _sweep_leftover(self, entry: Path) -> None:
        """Remove one `.tmp-*`/`.del-*` leftover, best-effort: one that can't even be
        stat'ed (a Windows ACL on drvfs), a directory that happens to carry such a name
        (not ours; unlink would fail on it anyway), or one still held open by another
        process is reported and left, never a reason to stop startup."""
        try:
            is_dir = stat.S_ISDIR(entry.lstat().st_mode)
        except FileNotFoundError:
            return
        except OSError as exc:
            self._cleanup_failed(entry, "stat", exc)
            return
        if is_dir:
            self._cleanup_failed(entry, "sweep", "is a directory")
            return
        try:
            os.unlink(entry)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._cleanup_failed(entry, "sweep", exc)

    def scan(self) -> ScanResult:
        """Load every valid `*.voice.json`, in filename order (so "sorts earlier" in
        the case-duplicate rule means exactly this iteration order); skip and report
        the rest; sweep `.tmp-*`/`.del-*` leftovers; count `.breeze` files (BC-29).

        Filesystem trouble with any one entry is reported and skipped, never raised:
        one bad entry must not stop the server from starting (BC-25). A `*.voice.json`
        that can't be stat'ed or isn't a regular file (a directory, say) is skipped
        like an invalid one, and its name reserved, since it still occupies that name
        on disk.

        Emits `voices.loaded`, one `voice.skipped` per rejected entry, and
        `voice.cleanup_failed` for a leftover it couldn't remove.
        """
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        voices: list[VoiceFile] = []
        skipped: list[SkippedVoiceFile] = []
        breeze_count = 0
        known: CaseInsensitiveNames[Path] = CaseInsensitiveNames()
        skipped_paths: CaseInsensitiveNames[Path] = CaseInsensitiveNames()
        # Every validly named entry seen so far, loaded or skipped (folded name ->
        # filename): data-model.md's case-duplicate rule is about files that sort
        # earlier, not only about files that loaded. An invalid Alice.voice.json still
        # holds the name, so a valid alice.voice.json after it is the duplicate.
        first_seen: dict[str, str] = {}

        def skip(entry: Path, reason: str, name: str | None) -> None:
            skipped.append(SkippedVoiceFile(file=entry.name, reason=reason, name=name))
            if name is not None:
                skipped_paths.set(name, entry)

        for entry in sorted(self.voices_dir.iterdir(), key=lambda p: p.name):
            fname = entry.name
            if fname.startswith((TMP_PREFIX, DEL_PREFIX)):
                self._sweep_leftover(entry)
                continue
            if fname.endswith(BREEZE_SUFFIX):
                breeze_count += 1
                continue
            if not fname.endswith(SUFFIX):
                continue

            stem = fname[: -len(SUFFIX)]
            name = stem if voice_file.is_valid_name(stem) else None
            if name is not None:
                earlier = first_seen.get(CaseInsensitiveNames.key(name))
                if earlier is not None:
                    skip(entry, f"case-duplicate of {earlier}", name)
                    continue
                first_seen[CaseInsensitiveNames.key(name)] = fname
            try:
                # stat() follows a symlink, as is_file() did: a link to a real voice
                # file still loads.
                is_regular = stat.S_ISREG(entry.stat().st_mode)
            except OSError as exc:
                skip(entry, str(exc), name)
                continue
            if not is_regular:
                skip(entry, "not a regular file", name)
                continue
            try:
                voice = voice_file.decode(
                    _read_bounded(entry),
                    expected_id=stem,
                    codebooks=self._codebooks,
                    codebook_size=self._codebook_size,
                    codec_fingerprint=self._codec_fingerprint,
                )
            except (OSError, VoiceFileError) as exc:
                skip(entry, str(exc), name)
                continue

            voices.append(voice)
            known.set(voice.id, entry)  # decode() checked id == stem, a valid name

        with self._lock:
            self._known = known
            self._skipped = skipped_paths

        self._events.emit(
            "voices.loaded", loaded=len(voices), skipped=len(skipped), breeze_ignored=breeze_count
        )
        for item in skipped:
            self._events.emit("voice.skipped", file=item.file, reason=item.reason)

        return ScanResult(voices=voices, skipped=skipped, breeze_count=breeze_count)

    # ------------------------------------------------------------------- writing

    def _name_taken_locked(self, name: str) -> bool:
        """Ignoring case: a loaded voice's id or a skipped entry's name. Caller holds
        `_lock`."""
        return self._known.taken(name) or self._skipped.taken(name)

    def create(self, voice: VoiceFile) -> VoiceFile:
        """Atomically write `voice` under `<voices_dir>/<voice.id>.voice.json`.

        Never overwrites, checked twice: first against the index (a loaded voice or a
        skipped entry with the same name, ignoring case), then on disk by the commit
        itself (`_commit_no_overwrite`), which refuses any existing file -- one added
        after the scan, or, on a case-insensitive filesystem, one differing only by
        case. Either refusal is `VoiceExists`. The whole create holds `_write_lock`.

        `created_at` is stamped here, from the injected clock, not taken from `voice` --
        it is this store's own record of when the write actually happened, not a
        caller-supplied guess. Returns the record as written (with that `created_at`).
        """
        with self._write_lock:
            with self._lock:
                if self._name_taken_locked(voice.id):
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
                _commit_no_overwrite(tmp_path, final_path, data)
            except FileExistsError as exc:
                raise VoiceExists(stamped.id) from exc
            finally:
                # After a link the temp name is a second name for the same file; after
                # the fallback or a failure it's a stray copy. Either way it goes, and
                # if it can't, the next scan's sweep takes it.
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self._cleanup_failed(tmp_path, "unlink_tmp", exc)
            _fsync_dir(self.voices_dir)
            with self._lock:
                self._known.set(stamped.id, final_path)
        return stamped

    # ------------------------------------------------------------------ deleting

    def remove(self, voice_id: str) -> bool:
        """Delete a loaded voice's file, or a skipped entry, named exactly `voice_id`;
        `False` if nothing matches.

        The lookup, the rename to `.del-<id>-<nonce>` and the index update all happen
        under `_write_lock`, so a concurrent `create()` of the same name waits until the
        old file is out of the way and can never have its own new file renamed away.
        The directory is fsynced after the rename, so a power loss can't bring the
        voice back (BC-28). If the rename fails (after retrying), the index is left
        alone and the failure propagates: the voice stays registered (DELETE's
        `500 voice_delete_failed`).

        A skipped entry that is a directory is refused the same way, with
        `IsADirectoryError`: deleting a directory tree nobody asked this server to
        write is too much for a DELETE to do, and renaming it to `.del-*` would only
        leave it for a sweep that won't remove directories either. It stays reserved
        until someone removes it by hand.

        Once the rename has succeeded the voice is gone, so the final unlink is
        best-effort: a failure is reported (`voice.cleanup_failed`) and the `.del-*`
        file left for the next scan's sweep, rather than turning a finished delete into
        a 500.
        """
        with self._write_lock:
            with self._lock:
                is_loaded = self._known.get(voice_id) is not None
                skipped_path = self._skipped.get(voice_id)
            if is_loaded:
                path = self.path_for(voice_id)
            elif skipped_path is not None:
                path = skipped_path
            else:
                return False

            trash = self.voices_dir / f"{DEL_PREFIX}{voice_id}-{self._nonce()}{SUFFIX}"
            try:
                if stat.S_ISDIR(os.lstat(path).st_mode):
                    raise IsADirectoryError(
                        errno.EISDIR, "a directory, not a voice file; remove it by hand", str(path)
                    )
                _rename_with_retry(path, trash, sleep=self._sleep)
                renamed = True
            except FileNotFoundError:
                renamed = False  # already gone from disk; only the index entry was left
            if renamed:
                _fsync_dir(self.voices_dir)
            with self._lock:
                if is_loaded:
                    self._known.pop(voice_id)
                else:
                    self._skipped.pop(voice_id)

        if renamed:
            try:
                os.unlink(trash)
            except OSError as exc:
                self._cleanup_failed(trash, "unlink", exc)
        return True
