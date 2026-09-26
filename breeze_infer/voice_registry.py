"""The in-memory voice registry: names, the unnamed cap, eviction and listing order.

data-model.md "Voice" ("Registry rules") plus BC-25, BC-26, BC-48. Sits above
`voice_store.py` (which owns the on-disk half of a saved voice) and knows nothing about
files itself -- `register_saved` is called once the store has already committed a
voice's file; `remove` only drops the in-memory entry, and the caller (T065's routes)
is responsible for also removing the file and the prefix-cache entry.

Unlike `api-alignment`'s `voice_index.py` (a single combined insertion order across
both tiers, one 64-entry cap over everything), this feature's cap and eviction apply
only to unnamed voices (BC-48), and listing order is two independently ordered
sequences concatenated -- saved sorted by id, then unnamed in registration order
(BC-25) -- so `_saved` and `_unnamed_order` are kept apart rather than merged into one
list. `api_record`/`voice_seconds` are borrowed from `A:breeze_infer/voice_index.py`
(~61-85) unchanged; the `A:` file's `memory_voice_id`/`fnv1a64` are not: this feature's
unnamed id is `unnamed_id` below (blake2b, not the C++-parity FNV-1a -- research.md R13,
the spec requires only the id *format*, not cross-server parity).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from breeze_infer.limits import UNNAMED_VOICE_CAP
from breeze_infer.voice_file import CaseInsensitiveNames, VoiceFile, prefix_key
from breeze_infer.voice_store import SkippedVoiceFile


def voice_seconds(frames: int, *, samples_per_frame: int, sample_rate: int) -> float:
    return round(frames * samples_per_frame / sample_rate, 2)


def api_record(
    *,
    voice_id: str,
    ref_text: str,
    frames: int,
    encode_ms: int,
    saved: bool,
    sample_rate: int,
    samples_per_frame: int,
) -> dict[str, Any]:
    """The exact `POST`/`GET /v1/voices` response shape (contracts/http-api.md)."""
    return {
        "id": voice_id,
        "frames": frames,
        "seconds": voice_seconds(
            frames, samples_per_frame=samples_per_frame, sample_rate=sample_rate
        ),
        "encode_ms": int(encode_ms),
        "saved": saved,
        "ref_text": ref_text,
    }


def unnamed_id(wav_bytes: bytes, text: str) -> str:
    """`"v_" + blake2b(len(wav) ‖ wav ‖ text, digest_size=8).hexdigest()`
    (data-model.md "Registry rules", research.md R13).

    `len(wav)` is a fixed-width (8-byte, big-endian) prefix, not the bytes' own varint
    or decimal spelling: without a fixed width, `wav=b"AB", text="C"` and
    `wav=b"A", text="BC"` would hash the identical byte string `b"ABC"` and collide.
    """
    digest = hashlib.blake2b(
        len(wav_bytes).to_bytes(8, "big") + wav_bytes + text.encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"v_{digest}"


class NameTaken(Exception):
    """`register_saved` found `voice.id` already claimed: by another saved voice
    (case-insensitively), by a name a skipped file still reserves, or because it falls
    in the `v_` namespace reserved for generated unnamed ids (BC-26)."""


@dataclass(frozen=True)
class SavedVoice:
    """A saved voice. Its codes are kept in memory, loaded at scan or registration, so a
    speech request resolves the voice without reading its file (T066): about 12 KB for a 30 s
    reference (376 frames x 16 codebooks, int16).

    `prefix_len` (the length its KV prefix builds to, `synthesis.measure_voice_prefix`) and
    `prefix_key` (its prefix cache key, `voice_file.prefix_key`) are computed once, when the
    voice is registered or scanned, so no speech request re-measures or re-hashes them."""

    id: str
    ref_text: str
    codes: np.ndarray
    frames: int
    encode_ms: int
    prefix_len: int
    prefix_key: tuple[str, str]

    @classmethod
    def from_file(cls, voice: VoiceFile, prefix_len: int) -> SavedVoice:
        return cls(
            id=voice.id,
            ref_text=voice.ref_text,
            codes=voice.codes,
            frames=voice.frames,
            encode_ms=voice.encode_ms,
            prefix_len=prefix_len,
            prefix_key=prefix_key(voice.id, voice.ref_text, voice.codes),
        )


@dataclass(frozen=True)
class MemoryVoice:
    """An unnamed voice: codes live only in memory, for this process's life. `prefix_len` and
    `prefix_key` as for `SavedVoice`."""

    id: str
    ref_text: str
    codes: np.ndarray
    frames: int
    encode_ms: int
    prefix_len: int
    prefix_key: tuple[str, str]


@dataclass(frozen=True)
class ResolvedVoice:
    """What a speech request needs from a voice, either tier (`VoiceRegistry.lookup`)."""

    id: str
    ref_text: str
    codes: np.ndarray  # [frames, codebooks]
    prefix_len: int
    prefix_key: tuple[str, str]


@dataclass(frozen=True)
class VoiceEntry:
    """One voice as `GET`/`POST /v1/voices` describes it, either tier, no codes."""

    id: str
    ref_text: str
    frames: int
    encode_ms: int
    saved: bool


@dataclass(frozen=True)
class RemovedVoice:
    """What `remove` took out of the registry -- enough for the caller (T065) to know
    whether a file and/or a prefix-cache entry also needs dropping."""

    id: str
    kind: str  # "saved" | "unnamed" | "reserved"


def _entry(voice: SavedVoice | MemoryVoice, *, saved: bool) -> VoiceEntry:
    return VoiceEntry(
        id=voice.id, ref_text=voice.ref_text, frames=voice.frames, encode_ms=voice.encode_ms, saved=saved
    )


class VoiceRegistry:
    """In-memory voices: a case-insensitive name index (saved ids plus reserved skipped
    names), the unnamed cap and eviction, and `GET /v1/voices`'s listing order.

    One lock, held only over the plain dict/list bookkeeping below -- never across I/O
    (a file write/rename), the GPU (an encode) or hashing (a record's `prefix_key` is computed
    before the lock is taken).
    """

    def __init__(self, *, cap: int = UNNAMED_VOICE_CAP) -> None:
        self._cap = int(cap)
        self._lock = threading.Lock()
        self._saved: dict[str, SavedVoice] = {}  # id (as saved) -> SavedVoice
        self._name_index: dict[str, str] = {}  # CaseInsensitiveNames.key(id) -> saved id
        # Skipped files' names (value: the skip reason). Taken ignoring case, but
        # released by the exact stem: the store deletes that exact file, and two
        # skipped files differing only by case each hold the name until both are gone.
        self._reserved: CaseInsensitiveNames[str] = CaseInsensitiveNames()
        self._unnamed: dict[str, MemoryVoice] = {}
        self._unnamed_order: list[str] = []  # registration order, oldest first

    # ------------------------------------------------------------------- naming

    def name_taken(self, name: str) -> bool:
        """Case-insensitive: true for an existing saved voice's id, a name a skipped
        file reserves, or any name in the `v_` namespace (BC-26) -- reserved outright,
        whether or not any specific unnamed id has actually been minted.

        Takes the lock: without it, a concurrent `remove` or `load_from_scan` could
        change the indexes mid-check."""
        with self._lock:
            return self._name_taken_locked(name)

    def _name_taken_locked(self, name: str) -> bool:
        """`name_taken`, for a caller already holding `_lock`."""
        if name[:2].lower() == "v_":
            return True
        return CaseInsensitiveNames.key(name) in self._name_index or self._reserved.taken(name)

    # ------------------------------------------------------------------- startup

    def load_from_scan(
        self, voices: Iterable[tuple[VoiceFile, int]], skipped: Iterable[SkippedVoiceFile]
    ) -> None:
        """Rebuild the saved-voice and reserved-name state from one
        `VoiceStore.scan()` result: each loaded voice with its measured prefix length, and
        the skipped files. Startup only: unnamed voices never survive a restart
        (data-model.md "Lifecycle"), so this also clears them.
        """
        saved = [SavedVoice.from_file(voice, prefix_len) for voice, prefix_len in voices]
        with self._lock:
            self._saved.clear()
            self._name_index.clear()
            self._reserved.clear()
            self._unnamed.clear()
            self._unnamed_order.clear()
            for voice in saved:
                self._saved[voice.id] = voice
                self._name_index[CaseInsensitiveNames.key(voice.id)] = voice.id
            for item in skipped:
                if item.name is not None:
                    self._reserved.set(item.name, item.reason)

    # --------------------------------------------------------------- registering

    def register_saved(self, voice: VoiceFile, *, prefix_len: int) -> VoiceEntry:
        """Add a voice the store has already committed to disk, with its measured prefix
        length. Raises `NameTaken` if `voice.id` is no longer free -- the caller's own
        commit-time re-check (T065) should have ruled this out already; this is the
        in-memory side of the same guarantee, not a second source of truth."""
        saved = SavedVoice.from_file(voice, prefix_len)
        with self._lock:
            if self._name_taken_locked(voice.id):
                raise NameTaken(voice.id)
            self._saved[voice.id] = saved
            self._name_index[CaseInsensitiveNames.key(voice.id)] = voice.id
            return _entry(saved, saved=True)

    def find_unnamed(self, voice_id: str) -> VoiceEntry | None:
        """The dedupe lookup for `POST`'s unnamed path (contract step 4): exact match,
        since an unnamed id is always the same 16 lowercase hex characters for the same
        audio and transcript."""
        with self._lock:
            voice = self._unnamed.get(voice_id)
            return _entry(voice, saved=False) if voice is not None else None

    def register_unnamed(
        self,
        *,
        id: str,
        ref_text: str,
        codes: np.ndarray,
        frames: int,
        encode_ms: int,
        prefix_len: int,
    ) -> tuple[VoiceEntry, str | None]:
        """Insert a new unnamed voice, evicting the oldest unnamed one first if the cap
        (BC-48: saved voices never count) is already reached. An id already present is
        returned unchanged (FR-017: "identical unnamed registration ... returns the
        existing entry"), with no eviction -- it's not a new entry.

        Returns `(entry, evicted_id)`, so a caller can drop `evicted_id` from the
        prefix cache too.
        """
        key = prefix_key(id, ref_text, codes)
        with self._lock:
            existing = self._unnamed.get(id)
            if existing is not None:
                return _entry(existing, saved=False), None

            evicted = None
            if len(self._unnamed_order) >= self._cap:
                evicted = self._unnamed_order.pop(0)
                del self._unnamed[evicted]

            voice = MemoryVoice(
                id=id,
                ref_text=ref_text,
                codes=codes,
                frames=frames,
                encode_ms=encode_ms,
                prefix_len=prefix_len,
                prefix_key=key,
            )
            self._unnamed[id] = voice
            self._unnamed_order.append(id)
            return _entry(voice, saved=False), evicted

    def lookup(self, voice_id: str) -> ResolvedVoice | None:
        """The voice a speech request names, or `None` if there is none. Exact, as `DELETE`
        is: case is ignored only at create (data-model.md "Registry rules")."""
        with self._lock:
            voice = self._saved.get(voice_id) or self._unnamed.get(voice_id)
            if voice is None:
                return None
            return ResolvedVoice(
                id=voice.id,
                ref_text=voice.ref_text,
                codes=voice.codes,
                prefix_len=voice.prefix_len,
                prefix_key=voice.prefix_key,
            )

    # -------------------------------------------------------------------- removing

    def remove(self, voice_id: str) -> RemovedVoice | None:
        """Drop `voice_id` from whichever tier holds it (or from the reserved-name
        set, for a skipped file's id). `None` if it's unknown. Only the in-memory
        registry changes here -- the caller also removes the file (if any) and the
        prefix-cache entry.

        Every match is exact: DELETE takes the id as `GET /v1/voices` lists it (what
        the SillyTavern extension sends back), or a skipped file's exact stem. Only
        uniqueness at create ignores case. `VoiceStore.remove` follows the same rule,
        so the two never disagree about which id a DELETE names."""
        with self._lock:
            if voice_id in self._saved:
                del self._saved[voice_id]
                self._name_index.pop(CaseInsensitiveNames.key(voice_id), None)
                return RemovedVoice(id=voice_id, kind="saved")
            if voice_id in self._unnamed:
                del self._unnamed[voice_id]
                self._unnamed_order.remove(voice_id)
                return RemovedVoice(id=voice_id, kind="unnamed")
            if self._reserved.pop(voice_id) is not None:
                return RemovedVoice(id=voice_id, kind="reserved")
            return None

    # -------------------------------------------------------------------- listing

    def list_records(self, *, sample_rate: int, samples_per_frame: int) -> list[dict[str, Any]]:
        """BC-25: saved voices sorted by id, then unnamed voices in registration
        order."""
        with self._lock:
            saved = [self._saved[key] for key in sorted(self._saved)]
            unnamed = [self._unnamed[key] for key in self._unnamed_order]
        records = [
            api_record(
                voice_id=v.id,
                ref_text=v.ref_text,
                frames=v.frames,
                encode_ms=v.encode_ms,
                saved=True,
                sample_rate=sample_rate,
                samples_per_frame=samples_per_frame,
            )
            for v in saved
        ]
        records += [
            api_record(
                voice_id=v.id,
                ref_text=v.ref_text,
                frames=v.frames,
                encode_ms=v.encode_ms,
                saved=False,
                sample_rate=sample_rate,
                samples_per_frame=samples_per_frame,
            )
            for v in unnamed
        ]
        return records

    def unnamed_count(self) -> int:
        with self._lock:
            return len(self._unnamed_order)

    def saved_count(self) -> int:
        with self._lock:
            return len(self._saved)
