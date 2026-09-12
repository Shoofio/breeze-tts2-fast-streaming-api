from __future__ import annotations

from pathlib import Path

from breeze_infer import compile_cache


def test_cli_value_wins_over_env(tmp_path: Path) -> None:
    env = {compile_cache.CACHE_DIR_ENV: str(tmp_path / "from-env")}
    chosen = compile_cache.resolve_cache_dir(
        tmp_path / "from-cli", environ=env, default=tmp_path / "default"
    )
    assert chosen == (tmp_path / "from-cli").resolve()
    assert env[compile_cache.CACHE_DIR_ENV] == str(chosen)
    assert chosen.is_dir()


def test_existing_env_is_respected_when_no_cli_value(tmp_path: Path) -> None:
    env = {compile_cache.CACHE_DIR_ENV: str(tmp_path / "launcher")}
    chosen = compile_cache.resolve_cache_dir(
        None, environ=env, default=tmp_path / "default"
    )
    assert chosen == (tmp_path / "launcher").resolve()
    assert not (tmp_path / "default").exists()


def test_default_used_when_nothing_is_set(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    chosen = compile_cache.resolve_cache_dir(
        None, environ=env, default=tmp_path / "default"
    )
    assert chosen == tmp_path / "default"
    assert chosen.is_dir()
    assert env[compile_cache.CACHE_DIR_ENV] == str(chosen)


def test_empty_env_value_falls_back_to_default(tmp_path: Path) -> None:
    env = {compile_cache.CACHE_DIR_ENV: ""}
    chosen = compile_cache.resolve_cache_dir(
        None, environ=env, default=tmp_path / "default"
    )
    assert chosen == tmp_path / "default"


def test_default_cache_dir_is_inside_the_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert compile_cache.DEFAULT_CACHE_DIR == repo_root / ".cache" / "torchinductor"


def test_phase_seconds_is_a_flat_float_dict() -> None:
    phases = compile_cache.compile_phase_seconds()
    assert all(isinstance(v, float) for v in phases.values())


def test_stats_have_stable_keys_and_integer_values() -> None:
    stats = compile_cache.compile_cache_stats()
    expected = {
        "inductor.fxgraph_cache_hit",
        "inductor.fxgraph_cache_miss",
        "inductor.fxgraph_cache_bypass",
        "aot_autograd.autograd_cache_hit",
        "aot_autograd.autograd_cache_miss",
        "aot_autograd.autograd_cache_bypass",
        "aot_autograd.autograd_cache_guard_miss",
        "stats.unique_graphs",
        "stats.calls_captured",
    }
    assert set(stats) == expected
    assert all(isinstance(value, int) for value in stats.values())


def test_diff_stats_subtracts_per_key() -> None:
    before = {"a": 1, "b": 5}
    after = {"a": 4, "b": 5, "c": 2}
    assert compile_cache.diff_stats(before, after) == {"a": 3, "b": 0, "c": 2}


def test_describe_reports_dir_and_counters(tmp_path: Path) -> None:
    block = compile_cache.describe(tmp_path)
    assert block["cache_dir"] == str(tmp_path)
    assert isinstance(block["phase_seconds"], dict)
    assert "inductor.fxgraph_cache_hit" in block["counters"]


class _FakeTorchKey:
    """Mimics torch_key: callable, with the prepopulation hook."""

    def __init__(self, value: bytes) -> None:
        self.value = value
        self.calls = 0
        self.set_to: bytes | None = None

    def __call__(self) -> bytes:
        self.calls += 1
        return self.value

    def set(self, value: bytes) -> None:
        self.set_to = value


def test_pin_torch_key_computes_then_reuses(tmp_path: Path) -> None:
    fake = _FakeTorchKey(b"\x01\x02")
    first = compile_cache.pin_torch_key(tmp_path, torch_key=fake, fingerprint="fp")
    assert first == "miss"
    assert fake.calls == 1
    assert (tmp_path / compile_cache.TORCH_KEY_FILE).is_file()

    again = _FakeTorchKey(b"\xff")
    second = compile_cache.pin_torch_key(tmp_path, torch_key=again, fingerprint="fp")
    assert second == "hit"
    assert again.calls == 0
    assert again.set_to == b"\x01\x02"


def test_pin_torch_key_recomputes_when_install_changes(tmp_path: Path) -> None:
    compile_cache.pin_torch_key(
        tmp_path, torch_key=_FakeTorchKey(b"\x01"), fingerprint="fp-old"
    )
    fresh = _FakeTorchKey(b"\x02")
    status = compile_cache.pin_torch_key(tmp_path, torch_key=fresh, fingerprint="fp-new")
    assert status == "miss"
    assert fresh.calls == 1
    assert fresh.set_to is None


def test_pin_torch_key_without_hook_is_unavailable(tmp_path: Path) -> None:
    status = compile_cache.pin_torch_key(
        tmp_path, torch_key=lambda: b"\x00", fingerprint="fp"
    )
    assert status == "unavailable"


def test_pin_torch_key_recovers_when_torch_already_computed_key(tmp_path: Path) -> None:
    """torch_key.set asserts its cache is empty; if something compiled first,
    fall through to the (memoized) real value instead of crashing."""

    class _AlreadyComputed(_FakeTorchKey):
        def set(self, value: bytes) -> None:
            raise AssertionError("cache already populated")

    compile_cache.pin_torch_key(
        tmp_path, torch_key=_FakeTorchKey(b"\x01"), fingerprint="fp"
    )
    live = _AlreadyComputed(b"\x09")
    status = compile_cache.pin_torch_key(tmp_path, torch_key=live, fingerprint="fp")
    assert status == "miss"
    assert live.calls == 1
    import json

    saved = json.loads((tmp_path / compile_cache.TORCH_KEY_FILE).read_text())
    assert saved["key"] == "09"


def test_torch_install_fingerprint_is_stable_and_covers_whole_package() -> None:
    a = compile_cache.torch_install_fingerprint()
    b = compile_cache.torch_install_fingerprint()
    assert a == b
    assert len(a) == 64


def test_torch_install_fingerprint_tracks_the_wheel_record(monkeypatch) -> None:
    base = compile_cache.torch_install_fingerprint()
    monkeypatch.setattr(compile_cache, "_wheel_record", lambda: "different RECORD")
    assert compile_cache.torch_install_fingerprint() != base


def test_torch_key_record_write_is_atomic(tmp_path: Path) -> None:
    compile_cache.pin_torch_key(
        tmp_path, torch_key=_FakeTorchKey(b"\x01"), fingerprint="fp"
    )
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
