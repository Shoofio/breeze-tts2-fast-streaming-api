"""Tests for breeze_infer/settings.py: the launch-configuration parser.

Covers build_parser()/settings_from_args() -- the new home for command-line
parsing (T016). Tests for runtime behavior (ApiSettings, configure_compile_cache,
_load_app, FastBreezeStreamingRuntime properties) stay in test_runtime_flags.py;
none of those exercise a parser, so nothing moved.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from breeze_infer.settings import build_parser, settings_from_args


def test_defaults(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])

    assert settings.model_path == tmp_path
    assert settings.host == "127.0.0.1"
    assert settings.port == 8080
    assert settings.ws_port == 8081
    assert settings.cors == ()
    assert settings.split_chars == 600
    assert settings.chunk_first == 1
    assert settings.chunk_max == 25
    assert settings.voices_dir == Path("voices")
    assert settings.fast_all is None
    assert settings.fast_text_encoder is False
    assert settings.fast_backbone_prefill is False
    assert settings.fast_backbone_decode is False
    assert settings.fast_depth_decoder is False
    assert settings.fast_codec is False
    assert settings.attn_implementation == "eager"
    assert settings.compile_cache_dir is None


def test_settings_is_frozen(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.port = 9090  # type: ignore[misc]


def test_host_and_port_overrides(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--host", "0.0.0.0", "--port", "9000"]
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    # ws_port still derives from the overridden port.
    assert settings.ws_port == 9001


def test_ws_port_explicit(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--ws-port", "9500"])
    assert settings.ws_port == 9500


def test_ws_port_disabled(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--ws-port", "disabled"])
    assert settings.ws_port is None


def test_ws_port_garbage_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--ws-port", "not-a-port"])
    assert "ws-port" in capsys.readouterr().err


def test_ws_port_equal_to_port_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", "9000", "--ws-port", "9000"])
    assert "differ" in capsys.readouterr().err


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_port_out_of_range_rejected(
    tmp_path: Path, port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--port", str(port)])
    assert "--port" in capsys.readouterr().err


@pytest.mark.parametrize("ws_port", [0, -1, 65536])
def test_ws_port_out_of_range_rejected(
    tmp_path: Path, ws_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--ws-port", str(ws_port)])
    assert "--ws-port" in capsys.readouterr().err


def test_cors_absent_is_off(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path)])
    assert settings.cors == ()


def test_cors_bare_flag_allows_any_origin(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--cors"])
    assert settings.cors == ("*",)


def test_cors_allowlist_is_trimmed(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--cors", " https://a.example , https://b.example "]
    )
    assert settings.cors == ("https://a.example", "https://b.example")


def test_cors_wildcard_mixed_with_others_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--cors", "*,https://a.example"])
    assert "*" in capsys.readouterr().err


def test_split_chars_default_and_override(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).split_chars == 600
    settings = settings_from_args([str(tmp_path), "--split-chars", "0"])
    assert settings.split_chars == 0


def test_negative_split_chars_rejected_not_clamped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), "--split-chars", "-1"])
    assert "--split-chars" in capsys.readouterr().err


def test_chunk_first_clamped_to_chunk_max(tmp_path: Path) -> None:
    settings = settings_from_args(
        [str(tmp_path), "--chunk-first", "100", "--chunk-max", "25"]
    )
    assert settings.chunk_first == 25
    assert settings.chunk_max == 25


@pytest.mark.parametrize("flag,value", [("--chunk-first", "0"), ("--chunk-max", "0")])
def test_chunk_sizes_below_one_rejected(
    tmp_path: Path, flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([str(tmp_path), flag, value])
    assert flag in capsys.readouterr().err


def test_voices_dir_override(tmp_path: Path) -> None:
    settings = settings_from_args([str(tmp_path), "--voices-dir", "my-voices"])
    assert settings.voices_dir == Path("my-voices")


def test_fast_all_tri_state(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).fast_all is None
    assert settings_from_args([str(tmp_path), "--fast-all"]).fast_all is True
    assert settings_from_args([str(tmp_path), "--no-fast-all"]).fast_all is False


def test_individual_fast_flags(tmp_path: Path) -> None:
    settings = settings_from_args(
        [
            str(tmp_path),
            "--fast-text-encoder",
            "--fast-backbone-prefill",
            "--fast-backbone-decode",
            "--fast-depth-decoder",
            "--fast-codec",
        ]
    )
    assert settings.fast_text_encoder is True
    assert settings.fast_backbone_prefill is True
    assert settings.fast_backbone_decode is True
    assert settings.fast_depth_decoder is True
    assert settings.fast_codec is True


def test_attn_implementation_choices(tmp_path: Path) -> None:
    assert settings_from_args([str(tmp_path)]).attn_implementation == "eager"
    settings = settings_from_args(
        [str(tmp_path), "--attn-implementation", "sdpa"]
    )
    assert settings.attn_implementation == "sdpa"


def test_attn_implementation_rejects_flash_attention_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        settings_from_args(
            [str(tmp_path), "--attn-implementation", "flash_attention_2"]
        )
    assert "--attn-implementation" in capsys.readouterr().err


def test_compile_cache_dir_override(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cc"
    settings = settings_from_args(
        [str(tmp_path), "--compile-cache-dir", str(cache_dir)]
    )
    assert settings.compile_cache_dir == cache_dir


def test_model_path_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        settings_from_args([])
    assert "model_path" in capsys.readouterr().err


def test_build_parser_returns_argument_parser() -> None:
    import argparse

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_settings_never_reads_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Principle III: env resolution (e.g. TORCHINDUCTOR_CACHE_DIR) is the
    composition root's job, not the parser's -- so an unset compile cache dir
    must stay None regardless of what's in the environment."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/should/not/be/read")
    settings = settings_from_args([str(tmp_path)])
    assert settings.compile_cache_dir is None


@pytest.mark.parametrize("value", ["", ",", " , "])
def test_cors_with_an_empty_allowlist_is_rejected(value):
    with pytest.raises(SystemExit):
        settings_from_args(["/models/breeze", "--cors", value])
