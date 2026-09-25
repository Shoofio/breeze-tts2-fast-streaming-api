from __future__ import annotations

import pytest

from breeze_infer.bench_api import RunResult, parse_args, summarize


def test_old_api_defaults_port_7860_and_omits_short_voice():
    config = parse_args(["--api", "old"])
    assert config.url == "http://127.0.0.1:7860"
    assert "short_voice" not in config.cases
    assert set(config.cases) == {"short_design", "medium_design", "short_inline"}


def test_new_api_defaults_port_8080_and_includes_short_voice():
    config = parse_args(["--api", "new"])
    assert config.url == "http://127.0.0.1:8080"
    assert "short_voice" in config.cases


def test_explicit_url_overrides_the_api_default():
    config = parse_args(["--api", "old", "--url", "http://example.test:9000"])
    assert config.url == "http://example.test:9000"


def test_short_voice_is_rejected_for_old_api():
    with pytest.raises(SystemExit):
        parse_args(["--api", "old", "--cases", "short_design,short_voice"])


def test_voice_id_flag_is_rejected_for_old_api():
    with pytest.raises(SystemExit):
        parse_args(["--api", "old", "--voice-id", "alice"])


def test_unknown_case_name_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--cases", "nonexistent_case"])


def test_runs_defaults_to_three():
    assert parse_args(["--api", "new"]).runs == 3


def test_summarize_reports_medians_over_successful_runs():
    # sample_rate=24000, s16le mono: 24000 * 2 bytes/sample = 1 second of audio.
    results = [
        RunResult(status=200, ttfa_s=0.10, wall_s=1.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=200, ttfa_s=0.20, wall_s=2.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=200, ttfa_s=0.30, wall_s=3.0, nbytes=24000 * 2, sample_rate=24000),
    ]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms_median"] == pytest.approx(200.0)
    assert summary["rtf_median"] == pytest.approx(2.0)
    assert summary["statuses"] == [200, 200, 200]


def test_summarize_excludes_failed_runs_from_the_medians():
    results = [
        RunResult(status=200, ttfa_s=0.10, wall_s=1.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=409, ttfa_s=None, wall_s=0.01, nbytes=0, sample_rate=24000),
        RunResult(status="truncated", ttfa_s=0.05, wall_s=0.5, nbytes=100, sample_rate=24000),
    ]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms_median"] == pytest.approx(100.0)
    assert summary["statuses"] == [200, 409, "truncated"]


def test_summarize_reports_none_when_every_run_fails():
    results = [RunResult(status=409, ttfa_s=None, wall_s=0.01, nbytes=0, sample_rate=24000)]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms_median"] is None
    assert summary["rtf_median"] is None
