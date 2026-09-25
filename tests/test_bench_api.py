from __future__ import annotations

import statistics
from pathlib import Path

import httpx
import pytest

from breeze_infer import bench_api
from breeze_infer.bench_api import (
    MEDIUM,
    SHORT,
    BenchConfig,
    RunResult,
    build_cases,
    delete_voice,
    load_reference,
    needs_reference,
    parse_args,
    post_and_drain,
    post_with_retry,
    register_voice,
    run_benchmark,
    summarize,
)

# The full literal from git show api-alignment:breeze_infer/bench_api.py.
MEDIUM_SOURCE = (
    "Streaming speech synthesis has to balance two goals that pull in different "
    "directions. Listeners want the first sound as soon as possible, but they "
    "also want the voice to stay consistent from the first sentence to the "
    "last. A good server starts talking quickly, keeps a steady pace, and "
    "never lets the character of the voice drift halfway through a paragraph. "
    "This medium length passage exists to measure exactly that trade-off."
)

UNUSED_REF_AUDIO = Path("/unused/ref.wav")  # only valid where needs_reference is False


class _BrokenStream(httpx.SyncByteStream):
    """A body that sends one chunk, then dies mid-stream (BC-17)."""

    def __iter__(self):
        yield b"partial-audio"
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    def close(self):
        pass


class _ReadErrorStream(httpx.SyncByteStream):
    """A body that dies mid-stream with a transport error other than RemoteProtocolError."""

    def __iter__(self):
        yield b"partial-audio"
        raise httpx.ReadError("connection reset by peer")

    def close(self):
        pass


# -- argument parsing -------------------------------------------------------


def test_api_defaults_to_new_so_the_quickstart_command_works_without_it():
    config = parse_args(["--url", "http://127.0.0.1:8080", "--runs", "3"])
    assert config.api == "new"


def test_old_api_defaults_port_7860_and_omits_short_voice():
    config = parse_args(["--api", "old"])
    assert config.url == "http://127.0.0.1:7860"
    assert "short_voice" not in config.cases
    assert set(config.cases) == {
        "short_design", "medium_design", "short_inline", "medium_inline",
    }


def test_new_api_defaults_port_8080_and_includes_short_voice():
    config = parse_args(["--api", "new"])
    assert config.url == "http://127.0.0.1:8080"
    assert "short_voice" in config.cases
    assert "medium_inline" in config.cases


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


def test_runs_defaults_to_ten():
    assert parse_args(["--api", "new"]).runs == 10


def test_runs_below_one_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--runs", "0"])


def test_warmup_defaults_to_three():
    assert parse_args(["--api", "new"]).warmup == 3


def test_warmup_below_zero_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--warmup", "-1"])


def test_warmup_zero_is_allowed():
    assert parse_args(["--api", "new", "--warmup", "0"]).warmup == 0


def test_cases_are_trimmed_and_empty_entries_dropped():
    config = parse_args(["--api", "new", "--cases", " short_design , medium_design ,, "])
    assert config.cases == ("short_design", "medium_design")


def test_duplicate_case_names_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--cases", "short_design,short_design"])


def test_blank_cases_selects_nothing_and_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--cases", ""])


def test_all_commas_cases_selects_nothing_and_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--api", "new", "--cases", " , , "])


# -- lazy reference loading --------------------------------------------------


def test_needs_reference_for_short_inline_but_not_bare_design_cases():
    assert needs_reference(("short_inline",), voice_id=None)
    assert not needs_reference(("short_design", "medium_design"), voice_id=None)


def test_needs_reference_for_medium_inline():
    assert needs_reference(("medium_inline",), voice_id=None)


def test_needs_reference_for_short_voice_only_when_it_must_register():
    assert needs_reference(("short_voice",), voice_id=None)
    assert not needs_reference(("short_voice",), voice_id="alice")


def test_load_reference_reads_the_sibling_txt_file_by_default(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake-wav-bytes")
    (tmp_path / "ref.txt").write_text("hello from the file\n")

    audio, text = load_reference(ref_audio, None)
    assert audio == b"fake-wav-bytes"
    assert text == "hello from the file"


def test_load_reference_empty_string_is_not_replaced_by_the_file(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake-wav-bytes")
    (tmp_path / "ref.txt").write_text("hello from the file")

    _audio, text = load_reference(ref_audio, "")
    assert text == ""


def test_load_reference_missing_file_is_a_clean_systemexit(tmp_path):
    with pytest.raises(SystemExit):
        load_reference(tmp_path / "missing.wav", None)


def test_medium_inline_uses_the_medium_text_with_the_inline_reference():
    cases = build_cases(("medium_inline",), b"wav-bytes", "the transcript", None)
    data, files = cases["medium_inline"]
    assert data["text"] == MEDIUM
    assert data["ref_text"] == "the transcript"
    assert files["ref_audio"][1] == b"wav-bytes"


def test_short_inline_still_uses_the_short_text():
    cases = build_cases(("short_inline",), b"wav-bytes", "the transcript", None)
    assert cases["short_inline"][0]["text"] == SHORT


# -- MEDIUM text (must match the api-alignment source exactly) --------------


def test_medium_matches_the_api_alignment_source_exactly():
    assert MEDIUM == MEDIUM_SOURCE


# -- summarize ----------------------------------------------------------------


def test_summarize_reports_medians_and_per_run_values():
    # sample_rate=24000, s16le mono: 24000 * 2 bytes/sample = 1 second of audio.
    results = [
        RunResult(status=200, ttfa_s=0.10, wall_s=1.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=200, ttfa_s=0.20, wall_s=2.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=200, ttfa_s=0.30, wall_s=3.0, nbytes=24000 * 2, sample_rate=24000),
    ]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms_median"] == pytest.approx(200.0)
    assert summary["rtf_median"] == pytest.approx(2.0)
    assert summary["audio_s_median"] == pytest.approx(1.0)
    assert summary["ttfa_ms"] == [pytest.approx(100.0), pytest.approx(200.0), pytest.approx(300.0)]
    assert summary["rtf"] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
    assert summary["statuses"] == [200, 200, 200]


def test_summarize_per_run_lists_line_up_with_statuses_using_none_for_failures():
    results = [
        RunResult(status=200, ttfa_s=0.10, wall_s=1.0, nbytes=24000 * 2, sample_rate=24000),
        RunResult(status=409, ttfa_s=None, wall_s=0.01, nbytes=0, sample_rate=24000),
        RunResult(status="truncated", ttfa_s=0.05, wall_s=0.5, nbytes=100, sample_rate=24000),
    ]
    summary = summarize("short_design", results)
    assert summary["statuses"] == [200, 409, "truncated"]
    assert summary["ttfa_ms"] == [pytest.approx(100.0), None, None]
    assert summary["rtf"] == [pytest.approx(1.0), None, None]
    assert summary["ttfa_ms_median"] == pytest.approx(100.0)
    assert summary["rtf_median"] == pytest.approx(1.0)


def test_summarize_reports_none_when_every_run_fails():
    results = [RunResult(status=409, ttfa_s=None, wall_s=0.01, nbytes=0, sample_rate=24000)]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms"] == [None]
    assert summary["rtf"] == [None]
    assert summary["ttfa_ms_median"] is None
    assert summary["rtf_median"] is None
    assert summary["audio_s_median"] is None
    assert summary["ttfa_ms_min"] is None
    assert summary["ttfa_ms_p25"] is None


def test_summarize_reports_ttfa_min_and_p25():
    # TTFA is bimodal on short_design (cache hit vs. miss); min/p25 surface
    # the fast end that the median alone can hide.
    ttfa_s_values = [0.05, 0.10, 0.20, 1.00]
    results = [
        RunResult(status=200, ttfa_s=v, wall_s=v + 1.0, nbytes=24000 * 2, sample_rate=24000)
        for v in ttfa_s_values
    ]
    summary = summarize("short_design", results)

    expected_ttfa_ms = [v * 1000 for v in ttfa_s_values]
    expected_p25 = statistics.quantiles(expected_ttfa_ms, n=4, method="inclusive")[0]
    assert summary["ttfa_ms_min"] == pytest.approx(50.0)
    assert summary["ttfa_ms_p25"] == pytest.approx(expected_p25)


def test_summarize_ttfa_min_and_p25_use_the_single_value_with_one_run():
    results = [RunResult(status=200, ttfa_s=0.075, wall_s=1.0, nbytes=24000 * 2, sample_rate=24000)]
    summary = summarize("short_design", results)
    assert summary["ttfa_ms_min"] == pytest.approx(75.0)
    assert summary["ttfa_ms_p25"] == pytest.approx(75.0)


# -- post_and_drain, against a mocked transport ------------------------------


def test_post_and_drain_reports_ttfa_and_bytes_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-sample-rate": "24000"}, content=b"\x00\x01" * 12000)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {"text": "hi"}, None)

    assert result.status == 200
    assert result.nbytes == 24000
    assert result.sample_rate == 24000
    assert result.ttfa_s is not None


def test_post_and_drain_truncated_after_status_was_received():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-sample-rate": "24000"}, stream=_BrokenStream())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "truncated"


def test_post_and_drain_read_error_mid_stream_is_also_truncated():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-sample-rate": "24000"}, stream=_ReadErrorStream())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "truncated"


def test_post_and_drain_error_before_any_response_is_received():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("connection reset")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "error:RemoteProtocolError"


def test_post_and_drain_connect_error_before_any_response_is_labeled_by_type():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "error:ConnectError"


def test_post_and_drain_bad_sample_rate_header_is_an_error_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-sample-rate": "banana"}, content=b"\x00\x01")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "error:bad_sample_rate"


def test_post_and_drain_zero_sample_rate_is_an_error_not_a_zero_division():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-sample-rate": "0"}, content=b"\x00\x01")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = post_and_drain(client, {}, None)

    assert result.status == "error:bad_sample_rate"


# -- post_with_retry / register_voice (409 busy) -----------------------------


def test_post_with_retry_retries_409_then_succeeds():
    statuses = iter([409, 409, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(statuses)
        content = b"\x00\x01" * 100 if code == 200 else b""
        return httpx.Response(code, headers={"x-sample-rate": "24000"}, content=content)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    sleeps = []
    result = post_with_retry(client, {}, None, sleep=sleeps.append)

    assert result.status == 200
    assert len(sleeps) == 2


def test_post_with_retry_gives_up_after_the_budget_and_reports_busy():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(409, headers={"x-sample-rate": "24000"}, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    # First call sets the deadline at 0 + budget; the second already exceeds it,
    # so the loop gives up without a real 60-second wait.
    clock = iter([0.0, 100.0])
    result = post_with_retry(client, {}, None, sleep=lambda s: None, now=lambda: next(clock))

    assert result.status == "busy"
    assert calls == 1


def test_register_voice_retries_409_then_succeeds():
    statuses = iter([409, 409, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(statuses)
        if code == 200:
            return httpx.Response(200, json={"id": "v_test123"})
        return httpx.Response(409, json={"error": "busy", "code": "busy"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    sleeps = []
    voice_id = register_voice(client, b"wav-bytes", "hello", sleep=sleeps.append)

    assert voice_id == "v_test123"
    assert len(sleeps) == 2


# -- run_benchmark: warm-up reporting and voice cleanup ----------------------


def test_run_benchmark_emits_warmup_failed_when_warmup_is_not_200(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, headers={"x-sample-rate": "24000"}, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=1,
        ref_audio=UNUSED_REF_AUDIO, ref_text=None, cases=("short_design",), voice_id=None,
    )
    run_benchmark(client, config, sleep=lambda s: None)

    events = [line for line in capsys.readouterr().out.splitlines()]
    assert any('"warmup_failed"' in line and "short_design" in line for line in events)


def test_run_benchmark_emits_one_warmup_failed_event_per_failed_warmup(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, headers={"x-sample-rate": "24000"}, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=3,
        ref_audio=UNUSED_REF_AUDIO, ref_text=None, cases=("short_design",), voice_id=None,
    )
    run_benchmark(client, config, sleep=lambda s: None)

    events = capsys.readouterr().out.splitlines()
    warmup_failed = [line for line in events if '"warmup_failed"' in line]
    assert len(warmup_failed) == 3


def test_run_benchmark_skips_warmup_entirely_when_warmup_is_zero(capsys):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, headers={"x-sample-rate": "24000"}, content=b"\x00\x01" * 100)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=2, warmup=0,
        ref_audio=UNUSED_REF_AUDIO, ref_text=None, cases=("short_design",), voice_id=None,
    )
    run_benchmark(client, config, sleep=lambda s: None)

    assert len(calls) == 2  # exactly the timed runs, no extra warm-up request
    events = capsys.readouterr().out.splitlines()
    assert not any('"warmup_failed"' in line for line in events)


def test_run_benchmark_deletes_a_voice_it_registered_itself(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake-wav-bytes")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/voices" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/voices" and request.method == "POST":
            return httpx.Response(200, json={"id": "v_test123"})
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(200, headers={"x-sample-rate": "24000"}, content=b"\x00\x01" * 100)
        if request.url.path == "/v1/voices/v_test123" and request.method == "DELETE":
            return httpx.Response(200, json={"deleted": "v_test123", "file_kept": False})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=1,
        ref_audio=ref_audio, ref_text="hello", cases=("short_voice",), voice_id=None,
    )
    run_benchmark(client, config, sleep=lambda s: None)

    assert ("DELETE", "/v1/voices/v_test123") in calls


def test_run_benchmark_does_not_delete_a_preexisting_deterministic_voice(tmp_path):
    """The unnamed id is a hash of wav+text: if another client already
    registered the exact same pair, POST returns that id with 200 (no new
    voice created). Deleting it would delete someone else's voice."""
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake-wav-bytes")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/voices" and request.method == "GET":
            return httpx.Response(200, json=[{"id": "v_existing"}])
        if request.url.path == "/v1/voices" and request.method == "POST":
            return httpx.Response(200, json={"id": "v_existing"})
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(200, headers={"x-sample-rate": "24000"}, content=b"\x00\x01" * 100)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=1,
        ref_audio=ref_audio, ref_text="hello", cases=("short_voice",), voice_id=None,
    )
    run_benchmark(client, config, sleep=lambda s: None)

    assert all(method != "DELETE" for method, _ in calls)


def test_run_benchmark_does_not_delete_a_voice_given_via_voice_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, headers={"x-sample-rate": "24000"}, content=b"\x00\x01" * 100)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=1,
        ref_audio=UNUSED_REF_AUDIO, ref_text=None, cases=("short_voice",), voice_id="alice",
    )
    run_benchmark(client, config, sleep=lambda s: None)

    assert all(method != "DELETE" for method, _ in calls)
    assert all(path != "/v1/voices" for _, path in calls)  # never lists or registers


def test_run_benchmark_still_cleans_up_if_a_later_case_blows_up(tmp_path, monkeypatch):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake-wav-bytes")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/voices" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/voices" and request.method == "POST":
            return httpx.Response(200, json={"id": "v_test123"})
        if request.url.path == "/v1/voices/v_test123" and request.method == "DELETE":
            return httpx.Response(200, json={"deleted": "v_test123", "file_kept": False})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthesis blew up")

    monkeypatch.setattr(bench_api, "build_cases", boom)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    config = BenchConfig(
        api="new", url="http://test", runs=1, warmup=1,
        ref_audio=ref_audio, ref_text="hello", cases=("short_voice",), voice_id=None,
    )

    with pytest.raises(RuntimeError):
        run_benchmark(client, config, sleep=lambda s: None)

    assert ("DELETE", "/v1/voices/v_test123") in calls


def test_delete_voice_swallows_a_failed_delete(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    delete_voice(client, "v_test123")  # must not raise

    assert "could not delete" in capsys.readouterr().err
