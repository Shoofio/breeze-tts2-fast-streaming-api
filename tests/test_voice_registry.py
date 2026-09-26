"""tests/test_voice_registry.py -- breeze_infer/voice_registry.py (T057, T063).

Pure: no filesystem, no GPU. `VoiceFile`/`SkippedVoiceFile` instances here are plain
values, not read from disk -- this module only cares about the in-memory bookkeeping
(names, the unnamed cap, eviction, and listing order) that sits above the store.
"""

from __future__ import annotations

import numpy as np
import pytest

from breeze_infer.voice_file import VoiceFile
from breeze_infer.voice_registry import NameTaken, VoiceRegistry, unnamed_id
from breeze_infer.voice_store import SkippedVoiceFile

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1920


def _saved_file(id: str, *, frames: int = 4, encode_ms: int = 100) -> VoiceFile:
    return VoiceFile(
        id=id,
        ref_text=f"text for {id}",
        frames=frames,
        codebooks=16,
        codes=np.zeros((frames, 16), dtype=np.int16),
        codes_sha256="unused",
        codec_fingerprint="unused",
        encode_ms=encode_ms,
        created_at="2026-09-24T20:15:00Z",
    )


# The registry stores whatever prefix length its caller measured; its value is arbitrary here.
PREFIX_LEN = 10


def _registry(cap: int = 64) -> VoiceRegistry:
    return VoiceRegistry(cap=cap)


def _register_unnamed(registry: VoiceRegistry, id: str, *, frames: int = 4, encode_ms: int = 1):
    return registry.register_unnamed(
        id=id,
        ref_text="hi",
        codes=np.zeros((frames, 16), dtype=np.int16),
        frames=frames,
        encode_ms=encode_ms,
        prefix_len=PREFIX_LEN,
    )


def _scanned(*voices: VoiceFile) -> list[tuple[VoiceFile, int]]:
    """Scanned voices as `load_from_scan` takes them: each with its measured prefix length."""
    return [(voice, PREFIX_LEN) for voice in voices]


# --------------------------------------------------------------------------- BC-26


def test_bc_26_names_are_unique_ignoring_case_and_v_prefix_is_reserved():
    registry = _registry()
    registry.register_saved(_saved_file("alice"), prefix_len=PREFIX_LEN)

    assert registry.name_taken("alice") is True
    assert registry.name_taken("Alice") is True
    assert registry.name_taken("ALICE") is True
    assert registry.name_taken("bob") is False

    with pytest.raises(NameTaken):
        registry.register_saved(_saved_file("ALICE"), prefix_len=PREFIX_LEN)

    # The whole v_ namespace is reserved, whether or not any such id has ever been
    # minted -- it's not tied to a specific registration.
    assert registry.name_taken("v_0000000000000000") is True
    assert registry.name_taken("V_ANYTHING") is True
    with pytest.raises(NameTaken):
        registry.register_saved(_saved_file("v_deadbeefdeadbeef"), prefix_len=PREFIX_LEN)

    # A name a skipped file reserves collides too, even though nothing was ever saved
    # under it (data-model.md: "including names reserved by skipped files").
    registry.load_from_scan(voices=[], skipped=[SkippedVoiceFile(file="carol.voice.json", reason="broken", name="carol")])
    assert registry.name_taken("Carol") is True
    with pytest.raises(NameTaken):
        registry.register_saved(_saved_file("carol"), prefix_len=PREFIX_LEN)


# --------------------------------------------------------------------------- BC-48


def test_bc_48_cap_counts_only_unnamed_voices():
    registry = _registry(cap=3)
    for i in range(10):
        registry.register_saved(_saved_file(f"saved-{i}"), prefix_len=PREFIX_LEN)
    assert registry.saved_count() == 10

    for i in range(3):
        _entry, evicted = _register_unnamed(registry, f"v_{i:016x}")
        assert evicted is None
    assert registry.unnamed_count() == 3

    # A cap this small would already have started evicting if saved voices counted
    # toward it too.
    _entry, evicted = _register_unnamed(registry, "v_0000000000000003")
    assert evicted is not None
    assert registry.unnamed_count() == 3
    assert registry.saved_count() == 10  # saved voices are never touched by eviction


def test_oldest_unnamed_voice_evicted_first():
    registry = _registry(cap=2)
    _register_unnamed(registry, "v_0000000000000000")
    _register_unnamed(registry, "v_0000000000000001")

    _entry, evicted = _register_unnamed(registry, "v_0000000000000002")

    assert evicted == "v_0000000000000000"
    remaining = {r["id"] for r in registry.list_records(sample_rate=SAMPLE_RATE, samples_per_frame=SAMPLES_PER_FRAME)}
    assert remaining == {"v_0000000000000001", "v_0000000000000002"}


def test_an_identical_unnamed_registration_returns_the_existing_entry():
    registry = _registry(cap=64)
    first, evicted_first = _register_unnamed(registry, "v_0000000000000000", encode_ms=42)
    second, evicted_second = _register_unnamed(registry, "v_0000000000000000", encode_ms=999)

    assert evicted_first is None
    assert evicted_second is None
    assert second == first  # the existing entry, not a re-encoded one
    assert second.encode_ms == 42
    assert registry.unnamed_count() == 1


# --------------------------------------------------------------------------- BC-25


def test_bc_25_list_order_is_saved_sorted_then_unnamed_by_registration():
    registry = _registry()
    # Saved, registered out of alphabetical order.
    registry.register_saved(_saved_file("zeta"), prefix_len=PREFIX_LEN)
    registry.register_saved(_saved_file("alpha"), prefix_len=PREFIX_LEN)
    registry.register_saved(_saved_file("mid"), prefix_len=PREFIX_LEN)
    # Unnamed, registered in a specific order that isn't alphabetical either.
    _register_unnamed(registry, "v_bbbbbbbbbbbbbbbb")
    _register_unnamed(registry, "v_aaaaaaaaaaaaaaaa")

    records = registry.list_records(sample_rate=SAMPLE_RATE, samples_per_frame=SAMPLES_PER_FRAME)
    ids = [r["id"] for r in records]

    assert ids == ["alpha", "mid", "zeta", "v_bbbbbbbbbbbbbbbb", "v_aaaaaaaaaaaaaaaa"]
    assert [r["saved"] for r in records] == [True, True, True, False, False]


def test_list_record_shape_matches_the_contract():
    registry = _registry()
    registry.register_saved(_saved_file("alice", frames=375, encode_ms=812), prefix_len=PREFIX_LEN)

    [record] = registry.list_records(sample_rate=SAMPLE_RATE, samples_per_frame=SAMPLES_PER_FRAME)

    assert record["id"] == "alice"
    assert record["frames"] == 375
    assert record["seconds"] == round(375 * SAMPLES_PER_FRAME / SAMPLE_RATE, 2)
    assert record["encode_ms"] == 812
    assert isinstance(record["encode_ms"], int)
    assert record["saved"] is True
    assert record["ref_text"] == "text for alice"


# --------------------------------------------------------------------- unnamed_id


def test_unnamed_id_is_deterministic_and_shaped_like_the_contract():
    first = unnamed_id(b"some wav bytes", "a transcript")
    second = unnamed_id(b"some wav bytes", "a transcript")
    assert first == second
    assert first.startswith("v_")
    assert len(first) == len("v_") + 16
    assert all(ch in "0123456789abcdef" for ch in first[2:])


def test_unnamed_id_disambiguates_where_wav_ends():
    # Without a fixed-width length prefix, concatenating differently-split wav/text
    # could collide on the same combined bytes.
    a = unnamed_id(b"AB", "C")
    b = unnamed_id(b"A", "BC")
    assert a != b


# ----------------------------------------------------------------------------- misc


def test_remove_drops_a_saved_voice_and_frees_its_name():
    registry = _registry()
    registry.register_saved(_saved_file("alice"), prefix_len=PREFIX_LEN)
    removed = registry.remove("alice")
    assert removed is not None
    assert removed.kind == "saved"
    assert registry.name_taken("alice") is False


def test_remove_drops_an_unnamed_voice():
    registry = _registry()
    _register_unnamed(registry, "v_0000000000000000")
    removed = registry.remove("v_0000000000000000")
    assert removed is not None
    assert removed.kind == "unnamed"
    assert registry.unnamed_count() == 0


def test_remove_of_unknown_id_returns_none():
    registry = _registry()
    assert registry.remove("nobody") is None


def test_load_from_scan_clears_unnamed_voices_but_not_saved():
    registry = _registry()
    registry.register_saved(_saved_file("alice"), prefix_len=PREFIX_LEN)
    _register_unnamed(registry, "v_0000000000000000")

    registry.load_from_scan(voices=_scanned(_saved_file("alice")), skipped=[])

    assert registry.unnamed_count() == 0
    assert registry.saved_count() == 1


# ------------------------------------------------- review finding #5: DELETE is exact


def test_remove_of_a_reserved_name_matches_the_skipped_file_stem_exactly():
    registry = _registry()
    registry.load_from_scan(
        voices=[], skipped=[SkippedVoiceFile(file="Carol.voice.json", reason="broken", name="Carol")]
    )
    assert registry.remove("carol") is None
    assert registry.name_taken("carol") is True  # still reserved, ignoring case
    removed = registry.remove("Carol")
    assert removed is not None and removed.kind == "reserved"
    assert registry.name_taken("carol") is False


def test_two_reserved_names_differing_only_by_case_are_released_separately():
    registry = _registry()
    registry.load_from_scan(
        voices=[],
        skipped=[
            SkippedVoiceFile(file="Carol.voice.json", reason="broken", name="Carol"),
            SkippedVoiceFile(file="carol.voice.json", reason="case-duplicate", name="carol"),
        ],
    )
    assert registry.remove("Carol") is not None
    assert registry.name_taken("CAROL") is True  # carol.voice.json still holds it
    assert registry.remove("carol") is not None
    assert registry.name_taken("CAROL") is False


@pytest.mark.parametrize("skipped", [False, True], ids=["saved", "skipped"])
def test_store_and_registry_agree_on_which_id_delete_matches(tmp_path, skipped: bool):
    """The routes (T065) call both; a DELETE for 'carol' against a file named Carol
    must be a miss in both, and 'Carol' a hit in both."""
    from datetime import datetime, timezone

    from breeze_infer import voice_file
    from breeze_infer.voice_store import VoiceStore
    from tests.fakes import RecordingEvents

    if skipped:
        (tmp_path / "Carol.voice.json").write_bytes(b"not json")
    else:
        (tmp_path / "Carol.voice.json").write_bytes(
            voice_file.encode(
                id="Carol",
                ref_text="hi",
                codes=np.zeros((4, 16), dtype=np.int16),
                codec_fingerprint="f" * 64,
                encode_ms=1,
                created_at="1970-01-01T00:00:00Z",
            )
        )
    store = VoiceStore(
        tmp_path,
        codebooks=16,
        codebook_size=2048,
        codec_fingerprint="f" * 64,
        events=RecordingEvents(),
        clock=lambda: datetime(2026, 9, 24, tzinfo=timezone.utc),
        sleep=lambda seconds: None,
    )
    scan = store.scan()
    registry = _registry()
    registry.load_from_scan(_scanned(*scan.voices), scan.skipped)

    assert store.remove("carol") is False
    assert registry.remove("carol") is None
    assert store.remove("Carol") is True
    assert registry.remove("Carol") is not None


# ------------------------------------------------------------------- review 38


def test_name_taken_holds_the_registry_lock():
    """#1: name_taken used to read the reserved names without the lock, so a
    concurrent remove() or load_from_scan() changed the dict mid-iteration ("dictionary
    changed size during iteration", a 500 on POST). Deterministic: while this test
    holds the lock, name_taken must not be able to finish."""
    import threading

    registry = _registry()
    registry.load_from_scan(
        voices=[], skipped=[SkippedVoiceFile(file="carol.voice.json", reason="broken", name="carol")]
    )
    finished = threading.Event()
    answer = {}

    def check():
        answer["taken"] = registry.name_taken("CAROL")
        finished.set()

    with registry._lock:
        worker = threading.Thread(target=check)
        worker.start()
        assert not finished.wait(timeout=0.3)
    assert finished.wait(timeout=10)
    worker.join()
    assert answer["taken"] is True


def test_case_insensitive_names_groups_exact_names_by_their_folded_key():
    """#9: the one case-folding structure both the store and the registry use."""
    from breeze_infer.voice_file import CaseInsensitiveNames

    names: CaseInsensitiveNames[str] = CaseInsensitiveNames()
    names.set("Carol", "first")
    names.set("carol", "second")
    assert names.taken("CAROL") and not names.taken("dave")
    assert names.get("Carol") == "first" and names.get("CAROL") is None
    assert names.pop("Carol") == "first"
    assert names.taken("CAROL")  # carol still holds it
    assert names.pop("carol") == "second"
    assert not names.taken("carol")
    assert names.pop("carol") is None


# ------------------------------------------------------------------- lookup (T066)


def test_lookup_resolves_saved_and_unnamed_voices_with_their_codes():
    """A speech request resolves a voice from memory alone: every record carries its codes
    and transcript (T066 option (a), loaded at scan or registration)."""
    registry = _registry()
    saved_codes = np.arange(4 * 16, dtype=np.int16).reshape(4, 16)
    unnamed_codes = np.ones((3, 16), dtype=np.int16)
    registry.load_from_scan(
        _scanned(VoiceFile(**{**_saved_file("scanned").__dict__, "codes": saved_codes})), []
    )
    registry.register_saved(_saved_file("alice"), prefix_len=PREFIX_LEN)
    registry.register_unnamed(
        id="v_0000000000000001",
        ref_text="hi",
        codes=unnamed_codes,
        frames=3,
        encode_ms=1,
        prefix_len=PREFIX_LEN,
    )

    scanned = registry.lookup("scanned")
    assert scanned is not None
    assert (scanned.id, scanned.ref_text) == ("scanned", "text for scanned")
    assert np.array_equal(scanned.codes, saved_codes)
    alice = registry.lookup("alice")
    assert alice is not None and alice.codes.shape == (4, 16)
    unnamed = registry.lookup("v_0000000000000001")
    assert unnamed is not None
    assert unnamed.ref_text == "hi"
    assert np.array_equal(unnamed.codes, unnamed_codes)


def test_lookup_is_exact_and_misses_removed_voices():
    registry = _registry()
    registry.register_saved(_saved_file("alice"), prefix_len=PREFIX_LEN)
    _register_unnamed(registry, "v_0000000000000001")

    assert registry.lookup("Alice") is None  # case is ignored only at create
    assert registry.lookup("nobody") is None
    registry.remove("alice")
    registry.remove("v_0000000000000001")
    assert registry.lookup("alice") is None
    assert registry.lookup("v_0000000000000001") is None


# ------------------------------------------------------------------- review 42 #7/#8


def test_each_record_carries_its_prefix_length_and_key_computed_once():
    """A speech request reads a voice's prefix length and cache key off its record: they are
    computed when the voice is registered or scanned, not per request."""
    from breeze_infer.voice_file import prefix_key

    registry = _registry()
    scanned = _saved_file("scanned")
    registry.load_from_scan([(scanned, 11)], [])
    registry.register_saved(_saved_file("alice"), prefix_len=12)
    registry.register_unnamed(
        id="v_0000000000000001",
        ref_text="hi",
        codes=np.ones((3, 16), dtype=np.int16),
        frames=3,
        encode_ms=1,
        prefix_len=13,
    )

    for voice_id, text, codes, length in [
        ("scanned", scanned.ref_text, scanned.codes, 11),
        ("alice", "text for alice", _saved_file("alice").codes, 12),
        ("v_0000000000000001", "hi", np.ones((3, 16), dtype=np.int16), 13),
    ]:
        voice = registry.lookup(voice_id)
        assert voice.prefix_len == length
        assert voice.prefix_key == prefix_key(voice_id, text, codes)
