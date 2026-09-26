"""tests/test_voice_store.py -- breeze_infer/voice_store.py (T056, T062).

Real `tmp_path`: this module's whole job is the filesystem, so faking it out would
test nothing. `sleep`, `clock` and `nonce` are injected so the retry/backoff and
timestamp/filename behaviour are deterministic and instant.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from breeze_infer import voice_file, voice_store
from breeze_infer.limits import MAX_VOICE_FILE_BYTES
from breeze_infer.voice_store import VoiceExists, VoiceStore
from tests.fakes import RecordingEvents

CODEBOOK_SIZE = 2048
CODEBOOKS = 16
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
        codebooks=CODEBOOKS,
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
        path.read_bytes(), expected_id="alice",
        codebooks=CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        codec_fingerprint=FINGERPRINT,
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
        codebooks=CODEBOOKS,
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


def test_remove_matches_the_id_exactly(tmp_path: Path):
    """Review finding #5: DELETE takes the id exactly as GET /v1/voices lists it (what
    the SillyTavern extension sends back); only create's uniqueness ignores case. The
    registry's remove matches exactly too, so the two can't disagree."""
    store = _store(tmp_path)
    store.create(_voice("Carol"))
    assert store.remove("carol") is False
    assert (tmp_path / "Carol.voice.json").is_file()
    assert store.remove("Carol") is True
    assert not (tmp_path / "Carol.voice.json").exists()


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


# ---------------------------------------------------------------- review findings
#
# Helpers for writing files straight onto disk, bypassing create(): another process, a
# hand-copied file, or a file left behind by an older build.


def _write_valid(tmp_path: Path, voice_id: str, *, ref_text: str = "hi") -> Path:
    path = tmp_path / f"{voice_id}{voice_store.SUFFIX}"
    path.write_bytes(
        voice_file.encode(
            id=voice_id,
            ref_text=ref_text,
            codes=_codes(seed=1),
            codec_fingerprint=FINGERPRINT,
            encode_ms=1,
            created_at="1970-01-01T00:00:00Z",
        )
    )
    return path


def _cleanup_events(events: RecordingEvents) -> list[dict[str, object]]:
    return [fields for name, fields in events.calls if name == "voice.cleanup_failed"]


def _require_case_sensitive(tmp_path: Path) -> None:
    probe = tmp_path / "CaseProbe"
    probe.write_text("x")
    try:
        if (tmp_path / "caseprobe").exists():
            pytest.skip("tmp_path is on a case-insensitive filesystem; this case needs two files differing only by case")
    finally:
        probe.unlink()


# ---- #2: create never overwrites, enforced on disk


def test_create_refuses_a_file_added_after_the_scan(tmp_path: Path):
    store = _store(tmp_path)
    store.scan()
    _write_valid(tmp_path, "alice", ref_text="from outside")

    with pytest.raises(VoiceExists):
        store.create(_voice("alice", ref_text="mine"))

    assert json_ref_text(tmp_path / "alice.voice.json") == "from outside"
    assert [p.name for p in tmp_path.iterdir()] == ["alice.voice.json"]  # no temp file left


def test_create_refuses_the_name_of_a_skipped_file(tmp_path: Path):
    (tmp_path / "bob.voice.json").write_bytes(b"not json")
    store = _store(tmp_path)
    store.scan()

    with pytest.raises(VoiceExists):
        store.create(_voice("bob"))
    assert (tmp_path / "bob.voice.json").read_bytes() == b"not json"


def test_create_refuses_a_case_variant_of_a_skipped_file(tmp_path: Path):
    _require_case_sensitive(tmp_path)
    (tmp_path / "Bob.voice.json").write_bytes(b"not json")
    store = _store(tmp_path)
    store.scan()

    with pytest.raises(VoiceExists):
        store.create(_voice("bob"))
    assert not (tmp_path / "bob.voice.json").exists()


def test_create_without_hardlinks_still_never_overwrites(tmp_path: Path, monkeypatch):
    """Where os.link isn't supported (EPERM on FAT/exFAT, for example), create falls
    back to an exclusive create of the final file: still no overwrite."""
    def no_links(src, dst):
        raise OSError(errno.EPERM, "simulated: hard links not supported")

    monkeypatch.setattr(os, "link", no_links)
    store = _store(tmp_path)
    store.scan()

    store.create(_voice("alice", ref_text="first"))
    assert json_ref_text(tmp_path / "alice.voice.json") == "first"

    _write_valid(tmp_path, "bob", ref_text="from outside")
    with pytest.raises(VoiceExists):
        store.create(_voice("bob", ref_text="mine"))
    assert json_ref_text(tmp_path / "bob.voice.json") == "from outside"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["alice.voice.json", "bob.voice.json"]


def json_ref_text(path: Path) -> str:
    return voice_file.decode(
        path.read_bytes(),
        expected_id=path.name[: -len(voice_store.SUFFIX)],
        codebooks=CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        codec_fingerprint=FINGERPRINT,
    ).ref_text


# ---- #3: remove deletes a skipped file and frees its name


def test_remove_deletes_a_skipped_file_and_frees_its_name(tmp_path: Path):
    (tmp_path / "bob.voice.json").write_bytes(b"not json")
    store = _store(tmp_path)
    store.scan()

    assert store.remove("bob") is True
    assert not (tmp_path / "bob.voice.json").exists()

    store.create(_voice("bob"))
    assert json_ref_text(tmp_path / "bob.voice.json") == "hello"


def test_remove_of_a_skipped_file_matches_its_stem_exactly(tmp_path: Path):
    (tmp_path / "Carol.voice.json").write_bytes(b"not json")
    store = _store(tmp_path)
    store.scan()

    assert store.remove("carol") is False
    assert (tmp_path / "Carol.voice.json").exists()
    assert store.remove("Carol") is True
    assert not (tmp_path / "Carol.voice.json").exists()


# ---- #4: a remove can never delete a concurrent create's file


def test_remove_cannot_delete_a_concurrent_create_of_the_same_name(tmp_path: Path):
    """Pauses remove() inside path_for() -- after its lookup, before its rename -- and
    tries to create the same name meanwhile. The create must either wait for the remove
    to finish or be refused; whichever, a create that returns has a file on disk at the
    end. The join timeout only decides how long the create gets to slip in; it can't
    change the outcome once remove holds the write lock across lookup and rename."""
    store = _store(tmp_path)
    store.create(_voice("alice", ref_text="old"))

    remove_paused = threading.Event()
    let_remove_go = threading.Event()
    real_path_for = store.path_for
    remover = {}

    def pausing_path_for(voice_id: str) -> Path:
        if threading.current_thread() is remover.get("thread") and not remove_paused.is_set():
            remove_paused.set()
            assert let_remove_go.wait(timeout=10)
        return real_path_for(voice_id)

    store.path_for = pausing_path_for  # type: ignore[method-assign]

    results: dict[str, object] = {}

    def do_remove():
        results["remove"] = store.remove("alice")

    def do_create():
        try:
            results["create"] = store.create(_voice("alice", ref_text="new"))
        except VoiceExists as exc:
            results["create"] = exc

    remover["thread"] = threading.Thread(target=do_remove)
    remover["thread"].start()
    assert remove_paused.wait(timeout=10)

    creator = threading.Thread(target=do_create)
    creator.start()
    creator.join(timeout=0.5)  # give the create every chance to run inside the gap
    let_remove_go.set()
    remover["thread"].join(timeout=10)
    creator.join(timeout=10)

    assert results["remove"] is True
    assert isinstance(results["create"], voice_file.VoiceFile)  # the name was free once remove ran
    assert json_ref_text(tmp_path / "alice.voice.json") == "new"


# ---- #6: after the rename, the unlink is best-effort, and the rename is fsynced


def test_remove_survives_an_unlink_failure_and_leaves_the_del_file_for_the_sweep(
    tmp_path: Path, monkeypatch
):
    events = RecordingEvents()
    store = _store(tmp_path, events=events)
    store.create(_voice("alice"))
    real_unlink = os.unlink

    def denied_for_del_files(path, *args, **kwargs):
        if Path(path).name.startswith(voice_store.DEL_PREFIX):
            raise PermissionError("simulated: a handle is still open")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", denied_for_del_files)
    assert store.remove("alice") is True

    assert not (tmp_path / "alice.voice.json").exists()
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert len(leftovers) == 1 and leftovers[0].startswith(voice_store.DEL_PREFIX)
    assert _cleanup_events(events) == [
        {"file": leftovers[0], "op": "unlink", "error": "simulated: a handle is still open"}
    ]

    monkeypatch.setattr(os, "unlink", real_unlink)
    _store(tmp_path).scan()  # the next start sweeps it
    assert list(tmp_path.iterdir()) == []


def test_remove_fsyncs_the_directory_after_the_rename(tmp_path: Path, monkeypatch):
    """BC-28: the rename to .del-* must reach the disk before remove returns, so a power
    loss can't bring the voice back."""
    store = _store(tmp_path)
    store.create(_voice("alice"))
    steps: list[str] = []
    real_replace, real_fsync_dir, real_unlink = os.replace, voice_store._fsync_dir, os.unlink

    def replace(src, dst):
        steps.append("rename")
        return real_replace(src, dst)

    def fsync_dir(path):
        steps.append("fsync_dir")
        return real_fsync_dir(path)

    def unlink(path, *args, **kwargs):
        steps.append("unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(voice_store, "_fsync_dir", fsync_dir)
    monkeypatch.setattr(os, "unlink", unlink)

    assert store.remove("alice") is True
    assert steps[:2] == ["rename", "fsync_dir"]


# ---- #8: the case-duplicate rule follows sort order over every file, valid or not


def test_a_case_duplicate_of_an_earlier_invalid_file_is_skipped(tmp_path: Path):
    _require_case_sensitive(tmp_path)
    (tmp_path / "Alice.voice.json").write_bytes(b"not json")  # sorts first ('A' < 'a')
    _write_valid(tmp_path, "alice")
    store = _store(tmp_path)

    result = store.scan()

    assert result.voices == []
    by_file = {item.file: item for item in result.skipped}
    assert set(by_file) == {"Alice.voice.json", "alice.voice.json"}
    assert "case-duplicate of Alice.voice.json" in by_file["alice.voice.json"].reason
    assert by_file["Alice.voice.json"].name == "Alice"  # the earlier file reserves the name


# ---- #9: filesystem trouble during scan is logged and skipped, never fatal


def test_scan_skips_a_leftover_directory(tmp_path: Path):
    (tmp_path / ".tmp-x").mkdir()
    _write_valid(tmp_path, "alice")
    events = RecordingEvents()
    result = _store(tmp_path, events=events).scan()

    assert [v.id for v in result.voices] == ["alice"]
    assert (tmp_path / ".tmp-x").is_dir()
    assert _cleanup_events(events) == [{"file": ".tmp-x", "op": "sweep", "error": "is a directory"}]


def test_scan_survives_a_leftover_it_cannot_unlink(tmp_path: Path, monkeypatch):
    (tmp_path / ".del-bob-n1.voice.json").write_text("garbage")
    _write_valid(tmp_path, "alice")
    real_unlink = os.unlink

    def denied(path, *args, **kwargs):
        if Path(path).name.startswith(voice_store.DEL_PREFIX):
            raise PermissionError("simulated: a handle is still open")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", denied)
    events = RecordingEvents()
    result = _store(tmp_path, events=events).scan()

    assert [v.id for v in result.voices] == ["alice"]
    assert (tmp_path / ".del-bob-n1.voice.json").exists()
    assert _cleanup_events(events) == [
        {"file": ".del-bob-n1.voice.json", "op": "sweep", "error": "simulated: a handle is still open"}
    ]


def test_scan_skips_a_file_larger_than_the_bound(tmp_path: Path):
    (tmp_path / "huge.voice.json").write_bytes(b" " * (MAX_VOICE_FILE_BYTES + 1))
    events = RecordingEvents()

    result = _store(tmp_path, events=events).scan()

    assert result.voices == []
    assert [item.file for item in result.skipped] == ["huge.voice.json"]
    assert f"larger than {MAX_VOICE_FILE_BYTES} bytes" in result.skipped[0].reason
    assert result.skipped[0].name == "huge"  # still reserves its name, like any skipped file
    assert ("voice.skipped", {"file": "huge.voice.json", "reason": result.skipped[0].reason}) in events.calls


def test_a_file_at_the_bound_is_read_and_judged_on_its_content(tmp_path: Path):
    """A file of exactly MAX_VOICE_FILE_BYTES is not skipped for its size: the
    padding keeps it valid JSON, so it loads."""
    path = _write_valid(tmp_path, "alice")
    data = path.read_bytes()
    path.write_bytes(data + b" " * (MAX_VOICE_FILE_BYTES - len(data)))

    result = _store(tmp_path).scan()

    assert [v.id for v in result.voices] == ["alice"]


def test_scan_reads_an_oversized_file_only_up_to_the_bound(tmp_path: Path):
    """A 64 MiB (sparse) file must not be read into memory whole: tracemalloc's peak
    over the scan stays far below the file's size."""
    with open(tmp_path / "huge.voice.json", "wb") as handle:
        handle.truncate(64 * 1024 * 1024)
    store = _store(tmp_path)

    tracemalloc.start()
    try:
        result = store.scan()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert [item.file for item in result.skipped] == ["huge.voice.json"]
    assert peak < 4 * MAX_VOICE_FILE_BYTES


# ------------------------------------------------------------------- review 38


def test_create_reports_a_temp_file_it_cannot_remove(tmp_path: Path, monkeypatch):
    """#4: the create itself succeeds; the stray temp file is reported and left for the
    next scan's sweep."""
    real_unlink = os.unlink

    def denied_for_tmp_files(path, *args, **kwargs):
        if Path(path).name.startswith(voice_store.TMP_PREFIX):
            raise PermissionError("simulated: a handle is still open")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", denied_for_tmp_files)
    events = RecordingEvents()
    store = _store(tmp_path, events=events)
    store.create(_voice("alice"))

    assert json_ref_text(tmp_path / "alice.voice.json") == "hello"
    assert _cleanup_events(events) == [
        {"file": ".tmp-alice-n0.voice.json", "op": "unlink_tmp", "error": "simulated: a handle is still open"}
    ]


def test_voice_store_no_longer_logs_through_the_stdlib():
    """#4 (Constitution VI): cleanup trouble is a named event, not a stray log line."""
    assert not hasattr(voice_store, "_log")


def _stat_denied_for(monkeypatch, names: set[str]) -> None:
    """Make stat() and lstat() of the named entries fail the way a Windows ACL on drvfs
    does: PermissionError with EACCES, which pathlib's is_file()/is_dir() re-raise."""
    real_stat, real_lstat = Path.stat, Path.lstat

    def stat_(self, *args, **kwargs):
        if self.name in names:
            raise PermissionError(errno.EACCES, "simulated: access denied", str(self))
        return real_stat(self, *args, **kwargs)

    def lstat_(self):
        if self.name in names:
            raise PermissionError(errno.EACCES, "simulated: access denied", str(self))
        return real_lstat(self)

    monkeypatch.setattr(Path, "stat", stat_)
    monkeypatch.setattr(Path, "lstat", lstat_)


def test_scan_skips_a_voice_file_it_cannot_stat_and_reserves_its_name(tmp_path: Path, monkeypatch):
    """#5"""
    _write_valid(tmp_path, "alice")
    _write_valid(tmp_path, "bob")
    _stat_denied_for(monkeypatch, {"bob.voice.json"})
    events = RecordingEvents()
    store = _store(tmp_path, events=events)

    result = store.scan()

    assert [v.id for v in result.voices] == ["alice"]
    assert [(item.file, item.name) for item in result.skipped] == [("bob.voice.json", "bob")]
    assert "simulated: access denied" in result.skipped[0].reason
    with pytest.raises(VoiceExists):
        store.create(_voice("bob"))


def test_scan_reports_a_leftover_it_cannot_stat(tmp_path: Path, monkeypatch):
    """#5"""
    (tmp_path / ".tmp-x").write_text("garbage")
    _write_valid(tmp_path, "alice")
    _stat_denied_for(monkeypatch, {".tmp-x"})
    events = RecordingEvents()

    result = _store(tmp_path, events=events).scan()

    assert [v.id for v in result.voices] == ["alice"]
    [event] = _cleanup_events(events)
    assert (event["file"], event["op"]) == (".tmp-x", "stat")
    assert "simulated: access denied" in str(event["error"])


@pytest.mark.parametrize("code", [errno.EIO, errno.ENOSPC, errno.EMLINK])
def test_create_does_not_fall_back_on_a_real_link_failure(tmp_path: Path, monkeypatch, code: int):
    """#6: only "links not supported" may take the non-atomic O_EXCL fallback; any other
    link failure is a failed write, and the temp file is still cleaned up."""
    def failing_link(src, dst):
        raise OSError(code, "simulated link failure")

    monkeypatch.setattr(os, "link", failing_link)
    store = _store(tmp_path)
    store.scan()

    with pytest.raises(OSError) as info:
        store.create(_voice("alice"))
    assert info.value.errno == code
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS])
def test_create_falls_back_when_links_are_not_supported(tmp_path: Path, monkeypatch, code: int):
    """#6"""
    def unsupported_link(src, dst):
        raise OSError(code, "simulated: links not supported")

    monkeypatch.setattr(os, "link", unsupported_link)
    store = _store(tmp_path)
    store.scan()

    store.create(_voice("alice"))
    assert json_ref_text(tmp_path / "alice.voice.json") == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["alice.voice.json"]


def test_a_directory_with_a_voice_file_name_is_skipped_and_reserved(tmp_path: Path):
    """#7: reported, its name reserved (so POST gets 409 from the index rather than an
    EEXIST at commit), and DELETE refuses it rather than deleting a directory tree."""
    (tmp_path / "bob.voice.json").mkdir()
    events = RecordingEvents()
    store = _store(tmp_path, events=events)

    result = store.scan()

    assert [(item.file, item.reason, item.name) for item in result.skipped] == [
        ("bob.voice.json", "not a regular file", "bob")
    ]
    assert ("voice.skipped", {"file": "bob.voice.json", "reason": "not a regular file"}) in events.calls
    with pytest.raises(VoiceExists):
        store.create(_voice("bob"))
    with pytest.raises(IsADirectoryError):
        store.remove("bob")
    assert (tmp_path / "bob.voice.json").is_dir()
    assert [p.name for p in tmp_path.iterdir()] == ["bob.voice.json"]  # not renamed to .del-*


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc to name an fd's file")
def test_the_exclusive_create_fallback_fsyncs_the_file(tmp_path: Path, monkeypatch):
    """#10: the fallback shares _write_bytes' write-flush-fsync helper, so the final
    file is fsynced exactly as the temp file is."""
    def no_links(src, dst):
        raise OSError(errno.EPERM, "simulated: hard links not supported")

    fsynced: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            fsynced.append(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[-1])
        return real_fsync(fd)

    monkeypatch.setattr(os, "link", no_links)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    _store(tmp_path).create(_voice("alice"))

    assert fsynced == [".tmp-alice-n0.voice.json", "alice.voice.json"]
