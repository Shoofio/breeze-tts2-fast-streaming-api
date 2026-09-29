"""Unit tests for the comparison logic of tests/live/cpp_examples.py (T080, SC-001).

No server: these cover how differences are found, how they are matched to EXPECTED_DIFFERENCES,
the report, the exit code, and the guard that keeps the harness away from real voices.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from tests.live import cpp_examples as cx
from tests.live.cpp_examples import (
    ANY,
    EXPECTED_DIFFERENCES,
    INT,
    MISSING,
    STRING,
    Difference,
    ExampleResult,
    ExpectedDifference,
    Match,
    compare,
    exit_code,
    explain,
    outcome,
    render_report,
    require_throwaway,
)

SPEC = Path(__file__).resolve().parents[1] / "specs" / "003-cpp-compatible-api" / "spec.md"


def diff(example: str, where: str, cpp: object, actual: object) -> Difference:
    return Difference(example, where, cpp, actual)


# ---- compare --------------------------------------------------------------------------------


def test_compare_equal_bodies_have_no_difference():
    assert compare("body", {"status": "ok", "sample_rate": 24000}, {"status": "ok", "sample_rate": 24000}) == []


def test_compare_reports_missing_and_extra_keys():
    diffs = compare("body", {"error": "busy"}, {"error": "busy", "code": "busy"})
    assert diffs == [("body.code", MISSING, "busy")]
    diffs = compare("body", {"error": "busy", "code": "busy"}, {"error": "busy"})
    assert diffs == [("body.code", "busy", MISSING)]


def test_compare_shapes_use_predicates():
    assert compare("body", {"frames": INT}, {"frames": 67}) == []
    assert compare("body", {"frames": INT}, {"frames": "67"}) == [("body.frames", INT, "67")]
    assert compare("body", {"frames": INT}, {"frames": True}) == [("body.frames", INT, True)]
    assert compare("body", {"frames": INT}, {}) == [("body.frames", INT, MISSING)]


def test_compare_is_type_strict_for_literals():
    assert compare("body.saved", True, 1) == [("body.saved", True, 1)]
    assert compare("body.sample_rate", 24000, 24000.0) == [("body.sample_rate", 24000, 24000.0)]


def test_compare_lists_element_wise_and_by_length():
    assert compare("events", [{"type": "done"}], [{"type": "done"}]) == []
    assert compare("events", [{"type": "done"}, {"type": "x"}], [{"type": "done"}, {"type": "y"}]) == [
        ("events.1.type", "x", "y")
    ]
    whole = compare("events", [{"type": "done"}], [])
    assert whole == [("events", [{"type": "done"}], [])]


# ---- explain / EXPECTED_DIFFERENCES ---------------------------------------------------------


def test_bc28_explains_file_kept_false_on_the_delete_example():
    assert explain(diff("cpp18", "body.file_kept", True, False)) == "BC-28"


def test_bc28_does_not_explain_the_same_field_elsewhere_or_other_values():
    assert explain(diff("cpp14", "body.file_kept", True, False)) is None
    assert explain(diff("cpp18", "body.file_kept", True, MISSING)) is None
    assert explain(diff("cpp18", "body.deleted", "st_live_tmp_cpp18", "x")) is None


def test_additive_error_code_is_explained_on_http_and_websocket():
    assert explain(diff("cpp08", "body.code", MISSING, "text_required")) == "ADD-2"
    assert explain(diff("cpp24", "events.0.code", MISSING, "unknown_voice")) == "ADD-2"
    assert explain(diff("cpp24", "events.0.request_type", MISSING, "start")) == "ADD-3"
    assert explain(diff("cpp24", "events.0.request_type", MISSING, None)) == "ADD-3"


def test_additive_entries_do_not_hide_a_changed_or_missing_value():
    # The C++ doc has the key but the value differs: not an addition.
    assert explain(diff("cpp08", "body.code", "x", "text_required")) is None
    # A missing documented key is never explained by the additive entries.
    assert explain(diff("cpp08", "body.error", "text is required", MISSING)) is None
    assert explain(diff("cpp08", "body.code", MISSING, 5)) is None


def test_unrelated_differences_are_unexplained():
    assert explain(diff("cpp03", "status", 200, 400)) is None
    assert explain(diff("cpp20", "events.1.text", "The harbour was quiet that morning.", "The harbour")) is None


def test_explain_uses_the_given_table_and_first_matching_entry():
    table = {
        "BC-01": ExpectedDifference("first", (Match("cpp0?", "status", cpp=200, actual=ANY),)),
        "BC-02": ExpectedDifference("second", (Match("*", "*"),)),
    }
    assert explain(diff("cpp03", "status", 200, 400), table) == "BC-01"
    assert explain(diff("cpp13", "status", 200, 400), table) == "BC-02"
    assert explain(diff("cpp03", "status", 200, 400), {}) is None


def test_expected_difference_keys_are_spec_ids():
    spec_bcs = set(re.findall(r"^\| (BC-\d\d) \|", SPEC.read_text(), flags=re.MULTILINE))
    assert len(spec_bcs) == 48
    for key, entry in EXPECTED_DIFFERENCES.items():
        assert re.fullmatch(r"BC-\d\d|ADD-\d", key), key
        if key.startswith("BC-"):
            assert key in spec_bcs, key
        assert entry.summary and entry.matches, key


# ---- outcome, exit code, report ------------------------------------------------------------


def result(example_id: str, *diffs: Difference, error: str | None = None, skipped: str | None = None) -> ExampleResult:
    return ExampleResult(example_id, f"doc > {example_id}", list(diffs), error=error, skipped=skipped)


def test_outcomes():
    assert outcome(result("cpp01")) == "PASS"
    assert outcome(result("cpp18", diff("cpp18", "body.file_kept", True, False))) == "EXPLAINED"
    assert outcome(result("cpp03", diff("cpp03", "status", 200, 400))) == "FAIL"
    assert outcome(result("cpp20", error="TimeoutError: no done")) == "FAIL"
    assert outcome(result("cpp12", skipped="needs --cors")) == "SKIP"


def test_exit_code_is_zero_when_every_difference_is_explained():
    results = [
        result("cpp01"),
        result("cpp18", diff("cpp18", "body.file_kept", True, False)),
        result("cpp08", diff("cpp08", "body.code", MISSING, "text_required")),
        result("cpp12", skipped="needs --cors"),
    ]
    assert exit_code(results) == 0


def test_exit_code_is_one_for_any_unexplained_difference():
    results = [
        result("cpp18", diff("cpp18", "body.file_kept", True, False), diff("cpp18", "status", 200, 500)),
    ]
    assert exit_code(results) == 1


def test_exit_code_is_one_when_an_example_errors():
    assert exit_code([result("cpp01"), result("cpp20", error="ConnectionRefusedError")]) == 1


def test_exit_code_of_an_empty_run_is_zero():
    assert exit_code([]) == 0


def test_report_names_the_bc_id_and_flags_unexplained():
    lines = render_report([
        result("cpp18", diff("cpp18", "body.file_kept", True, False)),
        result("cpp03", diff("cpp03", "status", 200, 400)),
        result("cpp20", error="TimeoutError: no done"),
    ])
    text = "\n".join(lines)
    assert "EXPLAINED cpp18" in text
    assert "body.file_kept: C++ True, here False  -> BC-28: DELETE removes" in text
    assert "status: C++ 200, here 400  -> UNEXPLAINED" in text
    assert "error: TimeoutError: no done" in text
    assert "3 examples: 0 pass, 1 explained, 2 fail, 0 skipped" in text
    assert "not seen this run: BC-18, ADD-2, ADD-3" in text
    assert lines[-1].startswith("RESULT: FAIL")


def test_report_flags_entries_not_yet_confirmed_live():
    table = {"BC-28": ExpectedDifference("file_kept", (Match("cpp18", "body.file_kept", True, False),), verify_live=True)}
    text = "\n".join(render_report([result("cpp18", diff("cpp18", "body.file_kept", True, False))], table))
    assert "-> BC-28, verify live: file_kept" in text
    assert "still marked verify live (confirm, then clear the flag): BC-28" in text


def test_report_ok_line():
    assert render_report([result("cpp01")])[-1] == "RESULT: OK"


# ---- voice safety and the example registry ---------------------------------------------------


@pytest.mark.parametrize("name", ["eric", "vale", "Eric", "VALE", "harbour", "v_0123456789abcdef", "st_live_tmp"])
def test_require_throwaway_refuses_real_voices(name):
    with pytest.raises(ValueError):
        require_throwaway(name)


def test_require_throwaway_allows_the_prefix_and_owned_ids():
    require_throwaway("st_live_tmp_cpp14")
    require_throwaway("v_0123456789abcdef", owned={"v_0123456789abcdef"})


def test_examples_cover_the_three_docs_with_unique_numbers():
    numbers = [e.number for e in cx.EXAMPLES]
    assert len(numbers) == len(set(numbers))
    assert {e.doc for e in cx.EXAMPLES} == {"server.md", "voices.md", "websocket.md"}
    for e in cx.EXAMPLES:
        assert len(f"{cx.TMP_PREFIX}{e.number:02d}_missing") <= 64  # a valid voice name


def test_example_numbers_cannot_repeat():
    with pytest.raises(ValueError):
        cx.example(1, "server.md", "duplicate")(lambda ctx, run: None)


def test_json_events_and_audio_ordering():
    items = [
        {"type": "ready"}, {"type": "started", "voice_id": ""}, {"type": "queued"},
        {"type": "speaking", "text": "a"}, cx.AUDIO, cx.AUDIO, {"type": "done"},
    ]
    assert cx.json_events(items) == [{"type": "started", "voice_id": ""}, {"type": "speaking", "text": "a"}, {"type": "done"}]
    assert cx.audio_after_each_speaking(items)
    assert not cx.audio_after_each_speaking([{"type": "speaking", "text": "a"}, {"type": "done"}])
    assert cx.collapse_audio(items[3:]) == [{"type": "speaking", "text": "a"}, cx.AUDIO, {"type": "done"}]
    assert STRING.test("") and not STRING.test(None)


# ---- sweep warnings ---------------------------------------------------------------------------


def sweep_ctx(handler) -> cx.Context:
    http = httpx.Client(base_url="http://server.test", transport=httpx.MockTransport(handler))
    return cx.Context(http=http, ref_wav=b"", ref_text="", cors=None, run_tag="t")


def test_sweep_warns_when_the_voice_list_is_not_200(capsys):
    ctx = sweep_ctx(lambda request: httpx.Response(503))
    assert cx.sweep(ctx) == []
    err = capsys.readouterr().err
    assert "WARNING" in err and "503" in err


def test_sweep_deletes_throwaways_and_stays_quiet_on_200_and_404(capsys):
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[{"id": "st_live_tmp_cpp14"}, {"id": "st_live_tmp_cpp15"}, {"id": "eric"}])
        deleted.append(request.url.path)
        return httpx.Response(404 if request.url.path.endswith("15") else 200)

    assert cx.sweep(sweep_ctx(handler)) == ["st_live_tmp_cpp14", "st_live_tmp_cpp15"]
    assert deleted == ["/v1/voices/st_live_tmp_cpp14", "/v1/voices/st_live_tmp_cpp15"]
    assert capsys.readouterr().err == ""


def test_sweep_warns_naming_a_voice_whose_delete_fails(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[{"id": "st_live_tmp_cpp14"}])
        return httpx.Response(500)

    assert cx.sweep(sweep_ctx(handler)) == ["st_live_tmp_cpp14"]
    err = capsys.readouterr().err
    assert "WARNING" in err and "st_live_tmp_cpp14" in err and "500" in err


def test_additive_codes_are_explained_on_prefixed_event_paths():
    # cpp23 checks `second.events`, so the path carries a prefix.
    assert explain(diff("cpp23", "second.events.2.code", MISSING, "unknown_voice")) == "ADD-2"
    assert explain(diff("cpp23", "second.events.2.request_type", MISSING, "start")) == "ADD-3"


def test_bc18_explains_options_without_cors_being_405_not_404():
    assert explain(diff("cpp12", "status", 404, 405)) == "BC-18"


def test_bc18_does_not_explain_other_statuses_or_examples():
    assert explain(diff("cpp12", "status", 404, 500)) is None
    assert explain(diff("cpp12", "status", 204, 405)) is None
    assert explain(diff("cpp13", "status", 404, 405)) is None
