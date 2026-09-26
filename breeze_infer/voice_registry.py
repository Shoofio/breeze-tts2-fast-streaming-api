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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from breeze_infer.limits import UNNAMED_VOICE_CAP
from breeze_infer.voice_file import VoiceFile
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
    id: str
    ref_text: str
    frames: int
    encode_ms: int

    @classmethod
    def from_file(cls, voice: VoiceFile) -> SavedVoice:
        return cls(id=voice.id, ref_text=voice.ref_text, frames=voice.frames, encode_ms=voice.encode_ms)


@dataclass(frozen=True)
class MemoryVoice:
    """An unnamed voice: codes live only in memory, for this process's life."""

    id: str
    ref_text: str
    codes: np.ndarray
    frames: int
    encode_ms: int


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
    (a file write/rename) or the GPU (an encode). `clock` is exposed for a caller timing
    an encode step (`encode_ms`) around the GPU work that must happen outside this
    lock; the registry itself never calls it.
    """

    def __init__(self, *, cap: int = UNNAMED_VOICE_CAP, clock: Callable[[], float]) -> None:
        self.clock = clock
        self._cap = int(cap)
        self._lock = threading.Lock()
        self._saved: dict[str, SavedVoice] = {}  # id (as saved) -> SavedVoice
        self._name_index: dict[str, str] = {}  # lowercased name -> saved id
        # exact name -> reason, for skipped files. Keyed exactly, not lowercased: DELETE
        # releases a skipped file's name by its exact stem (the store deletes that exact
        # file), and two skipped files differing only by case each hold the name until
        # both are deleted. name_taken still compares ignoring case.
        self._reserved: dict[str, str] = {}
        self._unnamed: dict[str, MemoryVoice] = {}
        self._unnamed_order: list[str] = []  # registration order, oldest first

    # ------------------------------------------------------------------- naming

    def name_taken(self, name: str) -> bool:
        """Case-insensitive: true for an existing saved voice's id, a name a skipped
        file reserves, or any name in the `v_` namespace (BC-26) -- reserved outright,
        whether or not any specific unnamed id has actually been minted."""
        if name[:2].lower() == "v_":
            return True
        lowered = name.lower()
        # A linear pass over the reserved names: only skipped files add them, a handful.
        return lowered in self._name_index or any(n.lower() == lowered for n in self._reserved)

    # ------------------------------------------------------------------- startup

    def load_from_scan(
        self, voices: Iterable[VoiceFile], skipped: Iterable[SkippedVoiceFile]
    ) -> None:
        """Rebuild the saved-voice and reserved-name state from one
        `VoiceStore.scan()` result. Startup only: unnamed voices never survive a
        restart (data-model.md "Lifecycle"), so this also clears them.
        """
        with self._lock:
            self._saved.clear()
            self._name_index.clear()
            self._reserved.clear()
            self._unnamed.clear()
            self._unnamed_order.clear()
            for voice in voices:
                self._saved[voice.id] = SavedVoice.from_file(voice)
                self._name_index[voice.id.lower()] = voice.id
            for item in skipped:
                if item.name is not None:
                    self._reserved[item.name] = item.reason

    # --------------------------------------------------------------- registering

    def register_saved(self, voice: VoiceFile) -> VoiceEntry:
        """Add a voice the store has already committed to disk. Raises `NameTaken` if
        `voice.id` is no longer free -- the caller's own commit-time re-check (T065)
        should have ruled this out already; this is the in-memory side of the same
        guarantee, not a second source of truth."""
        with self._lock:
            if self.name_taken(voice.id):
                raise NameTaken(voice.id)
            saved = SavedVoice.from_file(voice)
            self._saved[voice.id] = saved
            self._name_index[voice.id.lower()] = voice.id
            return _entry(saved, saved=True)

    def find_unnamed(self, voice_id: str) -> VoiceEntry | None:
        """The dedupe lookup for `POST`'s unnamed path (contract step 4): exact match,
        since an unnamed id is always the same 16 lowercase hex characters for the same
        audio and transcript."""
        with self._lock:
            voice = self._unnamed.get(voice_id)
            return _entry(voice, saved=False) if voice is not None else None

    def register_unnamed(
        self, *, id: str, ref_text: str, codes: np.ndarray, frames: int, encode_ms: int
    ) -> tuple[VoiceEntry, str | None]:
        """Insert a new unnamed voice, evicting the oldest unnamed one first if the cap
        (BC-48: saved voices never count) is already reached. An id already present is
        returned unchanged (FR-017: "identical unnamed registration ... returns the
        existing entry"), with no eviction -- it's not a new entry.

        Returns `(entry, evicted_id)`, so a caller can drop `evicted_id` from the
        prefix cache too.
        """
        with self._lock:
            existing = self._unnamed.get(id)
            if existing is not None:
                return _entry(existing, saved=False), None

            evicted = None
            if len(self._unnamed_order) >= self._cap:
                evicted = self._unnamed_order.pop(0)
                del self._unnamed[evicted]

            voice = MemoryVoice(id=id, ref_text=ref_text, codes=codes, frames=frames, encode_ms=encode_ms)
            self._unnamed[id] = voice
            self._unnamed_order.append(id)
            return _entry(voice, saved=False), evicted

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
                self._name_index.pop(voice_id.lower(), None)
                return RemovedVoice(id=voice_id, kind="saved")
            if voice_id in self._unnamed:
                del self._unnamed[voice_id]
                self._unnamed_order.remove(voice_id)
                return RemovedVoice(id=voice_id, kind="unnamed")
            if voice_id in self._reserved:
                del self._reserved[voice_id]
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
