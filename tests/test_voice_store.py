"""tests/test_voice_store.py -- breeze_infer/voice_store.py (T056, T062).

Real `tmp_path`: this module's whole job is the filesystem, so faking it out would
test nothing. `sleep`, `clock` and `nonce` are injected so the retry/backoff and
timestamp/filename behaviour are deterministic and instant.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from breeze_infer import voice_file, voice_store
from breeze_infer.voice_store import VoiceExists, VoiceStore
from tests.fakes import RecordingEvents

CODEBOOK_SIZE = 2048
FINGERPRINT = "f" * 64


def _codes(frames: int = 4, codebooks: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, CODEBOOK_SIZE, size=(frames, codebooks), dtype=np.int16)


def _voice(id: str = "alice", **overrides: object) -> voice_file.VoiceFile:
    defaults = {
        "id": id,
        "ref_text": "hello",
        "frames": 4,
        "codebooks": 16,
        "codes": _codes(),
        "codes_sha256": "unused-recomputed-by-encode",
        "codec_fingerprint": FINGERPRINT,
        "encode_ms": 100,
        "created_at": "1970-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return voice_file.VoiceFile(**defaults)


def _store(tmp_path: Path, *, events=None, clock=None, sleep=None, nonce=None) -> VoiceStore:
    counter = iter(range(10_000))
    return VoiceStore(
        tmp_path,
        codebook_size=CODEBOOK_SIZE,
        codec_fingerprint=FINGERPRINT,
        events=events if events is not None else RecordingEvents(),
        clock=clock if clock is not None else (lambda: datetime(2026, 9, 24, 20, 15, tzinfo=timezone.utc)),
        sleep=sleep if sleep is not None else (lambda seconds: None),
        nonce=nonce if nonce is not None else (lambda: f"n{next(counter)}"),
    )


# ------------------------------------------------------------------------- create


def test_create_writes_a_readable_file(tmp_path: Path):
    store = _store(tmp_path)
    written = store.create(_voice("alice"))

    assert written.created_at == "2026-09-24T20:15:00Z"  # from the injected clock
    path = tmp_path / "alice.voice.json"
    assert path.is_file()
    loaded = voice_file.decode(
        path.read_bytes(), expected_id="alice", codebook_size=CODEBOOK_SIZE, codec_fingerprint=FINGERPRINT
    )
    assert loaded.id == "alice"
    assert np.array_equal(loaded.codes, written.codes)


def test_create_never_overwrites_an_existing_file(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice", ref_text="first"))
    with pytest.raises(VoiceExists):
        store.create(_voice("alice", ref_text="second"))

    # the original file is untouched.
    loaded = voice_file.decode(
        (tmp_path / "alice.voice.json").read_bytes(),
        expected_id="alice",
        codebook_size=CODEBOOK_SIZE,
        codec_fingerprint=FINGERPRINT,
    )
    assert loaded.ref_text == "first"


def test_create_refuses_a_case_duplicate(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    with pytest.raises(VoiceExists):
        store.create(_voice("ALICE"))
    assert not (tmp_path / "ALICE.voice.json").exists()


def test_create_leaves_no_tmp_file_behind(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(voice_store.TMP_PREFIX)]
    assert leftovers == []


# ------------------------------------------------------------------------- remove


def test_remove_deletes_the_file(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    assert store.remove("alice") is True
    assert not (tmp_path / "alice.voice.json").exists()


def test_remove_is_case_insensitive(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    assert store.remove("ALICE") is True
    assert not (tmp_path / "alice.voice.json").exists()


def test_remove_of_an_unknown_id_returns_false(tmp_path: Path):
    store = _store(tmp_path)
    assert store.remove("nobody") is False


def test_remove_retries_five_times_on_permission_error_then_succeeds(tmp_path: Path, monkeypatch):
    store = _store(tmp_path)
    store.create(_voice("alice"))

    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < voice_store.DELETE_RENAME_ATTEMPTS:
            raise PermissionError("simulated: a handle is still open")
        return real_replace(src, dst)

    sleeps: list[float] = []
    monkeypatch.setattr(os, "replace", flaky_replace)
    store._sleep = sleeps.append  # the store's own injected sleep

    assert store.remove("alice") is True
    assert calls["count"] == voice_store.DELETE_RENAME_ATTEMPTS
    assert len(sleeps) == voice_store.DELETE_RENAME_ATTEMPTS - 1


def test_remove_gives_up_after_five_permission_errors(tmp_path: Path, monkeypatch):
    store = _store(tmp_path)
    store.create(_voice("alice"))

    def always_denied(src, dst):
        raise PermissionError("simulated: never released")

    monkeypatch.setattr(os, "replace", always_denied)
    store._sleep = lambda seconds: None

    with pytest.raises(PermissionError):
        store.remove("alice")
    # the failure is put back: the voice is still registered and the file still there.
    assert (tmp_path / "alice.voice.json").is_file()
    assert "alice" in store._known


# ------------------------------------------------------------------------------ scan


def test_scan_sweeps_leftover_tmp_and_del_files(tmp_path: Path):
    (tmp_path / ".tmp-alice-n0.voice.json").write_text("garbage")
    (tmp_path / ".del-bob-n1.voice.json").write_text("garbage")
    store = _store(tmp_path)

    result = store.scan()

    assert result.voices == []
    assert list(tmp_path.iterdir()) == []


def test_bc_29_breeze_files_are_ignored_and_counted(tmp_path: Path):
    (tmp_path / "legacy.breeze").write_bytes(b"whatever the C++ server wrote")
    (tmp_path / "another.breeze").write_bytes(b"more")
    store = _store(tmp_path)

    result = store.scan()

    assert result.breeze_count == 2
    assert result.voices == []
    # the .breeze files themselves are left alone -- only .tmp-/.del-* are swept.
    assert (tmp_path / "legacy.breeze").exists()


def test_bc_25_invalid_files_are_skipped_with_an_event(tmp_path: Path):
    (tmp_path / "broken.voice.json").write_text("not json")
    events = RecordingEvents()
    store = _store(tmp_path, events=events)

    result = store.scan()

    assert result.voices == []
    assert len(result.skipped) == 1
    assert result.skipped[0].file == "broken.voice.json"
    assert result.skipped[0].name == "broken"

    skip_events = [call for call in events.calls if call[0] == "voice.skipped"]
    assert len(skip_events) == 1
    assert skip_events[0][1]["file"] == "broken.voice.json"

    loaded_events = [call for call in events.calls if call[0] == "voices.loaded"]
    assert loaded_events == [("voices.loaded", {"loaded": 0, "skipped": 1, "breeze_ignored": 0})]


def test_a_skipped_file_with_a_valid_name_reserves_that_name(tmp_path: Path):
    """The file is broken (bad fingerprint), but its filename is still a syntactically
    valid voice name, so scan() reports it as reservable (voice_registry.py uses this
    to keep the name out of a fresh POST, and DELETE can still remove the file)."""
    store = _store(tmp_path)
    store.create(_voice("alice"))
    # Make the on-disk copy fail validation without touching its filename.
    broken = (tmp_path / "alice.voice.json").read_text().replace(FINGERPRINT, "0" * 64)
    (tmp_path / "alice.voice.json").write_text(broken)

    result = store.scan()

    assert result.voices == []
    assert len(result.skipped) == 1
    assert result.skipped[0].name == "alice"


def test_scan_skips_a_case_duplicate_of_an_earlier_file(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    # A second, case-differing, internally self-consistent file dropped directly on
    # disk (bypassing create()'s own guard) -- each file's own id matches its own
    # stem, so only the case comparison between the two filenames is at stake.
    other = voice_file.encode(
        id="ALICE",
        ref_text="hi",
        codes=_codes(seed=1),
        codec_fingerprint=FINGERPRINT,
        encode_ms=1,
        created_at="1970-01-01T00:00:00Z",
    )
    (tmp_path / "ALICE.voice.json").write_bytes(other)

    result = store.scan()

    # ASCII sorts "ALICE.voice.json" before "alice.voice.json" ('A' < 'a'), so ALICE
    # loads and alice.voice.json is skipped as its case-duplicate.
    assert [v.id for v in result.voices] == ["ALICE"]
    assert len(result.skipped) == 1
    assert result.skipped[0].file == "alice.voice.json"
    assert "case-duplicate" in result.skipped[0].reason


def test_scan_loads_a_valid_file_written_by_create(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))
    store.create(_voice("bob"))

    result = store.scan()

    assert sorted(v.id for v in result.voices) == ["alice", "bob"]
    assert result.skipped == []


def test_scan_rebuilds_the_known_index_so_create_still_refuses_a_duplicate(tmp_path: Path):
    store = _store(tmp_path)
    store.create(_voice("alice"))

    fresh = _store(tmp_path)  # a brand-new store, as if the process just restarted
    fresh.scan()
    with pytest.raises(VoiceExists):
        fresh.create(_voice("alice"))
