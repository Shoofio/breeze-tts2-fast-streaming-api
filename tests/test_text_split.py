"""Tests for the shared text segmenter (breeze_infer/text_split.py).

The C++ goldens in tests/cpp_golden/golden.json come from the real Breeze-TTS-2.cpp code (see
tests/cpp_golden/README.md). The port strips pieces and drops empty ones (research R11), so the
C++ pieces are normalized the same way before comparing. Every case that still differs is listed
in INTENTIONAL_DIFFERENCES with its new output and the breaking change that explains it; that
table is the evidence for SC-002.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from breeze_infer.text_split import segment, split_text, weigh

GOLDEN = json.loads(
    (Path(__file__).parent / "cpp_golden" / "golden.json").read_text(encoding="utf-8")
)
SPLIT_CASES = {c["name"]: c for c in GOLDEN["split_text"]}
DRAIN_CASES = {c["name"]: c for c in GOLDEN["drain"]}

# Case name -> (expected new output, BC id). A split case's output is its piece list; a drain
# case's output is (pieces, remaining).
INTENTIONAL_DIFFERENCES: dict[str, tuple[object, str]] = {
    # C++ treats a NUL after `.` as a closing quote (strchr matches the terminator), so `a.\0`
    # ended a sentence. NUL is rejected at validation now; here it is an ordinary character.
    "nul_byte_in_text": (["a.\x00 b", "c d e", "f g h", "i j k", "l m n"], "BC-46"),
    # C++ drain compared UTF-8 bytes (56 <= 60) and kept the buffer; by weight it is 82 > 60.
    "drain_byte_vs_weight_asymmetry_undrained": (
        (["привет как дела сегодня"], "хорошо"),
        "BC-39",
    ),
    # C++ drain didn't absorb the closing quote, so `."` never ended a sentence.
    "drain_no_quote_absorption": ((['He said "done."'], " next"), "BC-39"),
    # C++ drain's stop set left out the fullwidth period that split_text had.
    "drain_fullwidth_period_not_sentence_end": ((["Done．"], "next"), "BC-39"),
}


def _normalized(pieces: list[str]) -> list[str]:
    return [p.strip() for p in pieces if p.strip()]


def _split_output(case: dict) -> list[str]:
    return split_text(case["text"], budget=case["budget"], first_budget=case["first_budget"])


def _drain_output(case: dict) -> tuple[list[str], str]:
    return segment(case["buffer"], budget=case["budget"], final=case["force"])


def _cpp_output(name: str) -> object:
    if name in SPLIT_CASES:
        return _normalized(SPLIT_CASES[name]["result"])
    case = DRAIN_CASES[name]
    return _normalized(case["pieces"]), case["remaining"]


def _new_output(name: str) -> object:
    if name in SPLIT_CASES:
        return _split_output(SPLIT_CASES[name])
    return _drain_output(DRAIN_CASES[name])


# --- C++ goldens ---


@pytest.mark.parametrize(
    "name", [n for n in SPLIT_CASES if n not in INTENTIONAL_DIFFERENCES]
)
def test_split_text_matches_normalized_cpp_golden(name) -> None:
    assert _split_output(SPLIT_CASES[name]) == _cpp_output(name)


@pytest.mark.parametrize(
    "name", [n for n in DRAIN_CASES if n not in INTENTIONAL_DIFFERENCES]
)
def test_segment_matches_normalized_cpp_drain_golden(name) -> None:
    assert _drain_output(DRAIN_CASES[name]) == _cpp_output(name)


def test_intentional_differences_name_real_golden_cases() -> None:
    assert set(INTENTIONAL_DIFFERENCES) <= set(SPLIT_CASES) | set(DRAIN_CASES)
    assert {bc for _, bc in INTENTIONAL_DIFFERENCES.values()} <= {"BC-39", "BC-46"}


def _differences(bc: str) -> list[str]:
    return [n for n, (_, b) in INTENTIONAL_DIFFERENCES.items() if b == bc]


@pytest.mark.parametrize("name", _differences("BC-39"))
def test_bc_39_intentional_difference_from_cpp(name) -> None:
    """C++ drained with its own stop set (no closing quotes, no `．`/`…`) and measured the
    buffer in UTF-8 bytes; the shared segmenter uses one stop set and weighted length."""
    expected, _ = INTENTIONAL_DIFFERENCES[name]
    assert _new_output(name) == expected
    assert expected != _cpp_output(name), "no longer a difference; drop it from the table"


@pytest.mark.parametrize("name", _differences("BC-46"))
def test_bc_46_intentional_difference_from_cpp(name) -> None:
    """C++ counted a NUL byte as closing punctuation after `.!?;`."""
    expected, _ = INTENTIONAL_DIFFERENCES[name]
    assert _new_output(name) == expected
    assert expected != _cpp_output(name), "no longer a difference; drop it from the table"


ALL_GOLDEN_INPUTS = [
    pytest.param(c["text"], c["budget"], c["first_budget"], id=c["name"])
    for c in GOLDEN["split_text"]
] + [
    pytest.param(c["buffer"], c["budget"], 0, id=c["name"]) for c in GOLDEN["drain"]
]


@pytest.mark.parametrize("text,budget,first_budget", ALL_GOLDEN_INPUTS)
def test_bc_39_split_text_is_the_final_segment(text, budget, first_budget) -> None:
    """C++ had two segmenters (split_text for HTTP, drain for the WebSocket) that disagreed;
    now HTTP's split is exactly a final segment."""
    final_pieces, remaining = segment(text, budget=budget, first_budget=first_budget, final=True)
    assert final_pieces == split_text(text, budget=budget, first_budget=first_budget)
    assert remaining == ""


# --- streaming: a sentence end at the end of the buffer waits ---


def test_bc_39_number_split_across_messages_stays_whole() -> None:
    """C++ cut `3.` as soon as it was at the end of the buffer, so `3.14` sent as `3.` then `14`
    was spoken as two pieces."""
    pieces, buffer = segment("3.", budget=30, final=False)
    assert (pieces, buffer) == ([], "3.")
    pieces, buffer = segment(buffer + "14", budget=30, final=False)
    assert (pieces, buffer) == ([], "3.14")
    assert segment(buffer, budget=30, final=True) == (["3.14"], "")


def test_bc_39_sentence_end_at_buffer_end_waits_for_the_next_character() -> None:
    """C++ cut `Hi.` at once; it now waits for the next character, flush or end."""
    assert segment("Hi.", budget=30, final=False) == ([], "Hi.")
    assert segment("Hi. ", budget=30, final=False) == (["Hi."], " ")
    assert segment("Hi.", budget=30, final=True) == (["Hi."], "")
    # Closing quotes might still follow, so a quoted sentence end waits too.
    assert segment('He said "done."', budget=30, final=False) == ([], 'He said "done."')


def test_bc_39_cjk_stop_at_buffer_end_waits() -> None:
    """C++ cut after a CJK stop at the end of the buffer at once."""
    assert segment("好。", budget=30, final=False) == ([], "好。")
    assert segment("好。你", budget=30, final=False) == (["好。"], "你")
    assert segment("好。", budget=30, final=True) == (["好。"], "")


def test_bc_39_abbreviation_followed_by_a_space_is_cut_as_in_cpp() -> None:
    """Not a change: `Dr. Smith` is cut after `Dr.` on both interfaces, as C++ split_text did.
    Only a period at the very end of a non-final buffer waits."""
    assert split_text("Dr. Smith is here.", budget=15) == ["Dr.", "Smith is here."]
    assert segment("Dr. Smith is", budget=15, final=False) == (["Dr."], " Smith is")
    assert segment("Dr.", budget=15, final=False) == ([], "Dr.")


def test_bc_39_newline_cuts_on_the_websocket_too() -> None:
    """C++ drain's sentence_end ignored `\\n`, so line-separated text waited for the budget."""
    assert segment("first line\nsecond", budget=30, final=False) == (["first line"], "second")
    assert split_text("first line\nsecond line", budget=15) == ["first line", "second line"]


# --- budgets ---


def test_bc_39_opening_budget_applies_to_the_first_piece_only() -> None:
    """C++ drained the whole opening buffer against the 200 budget, so every piece of that
    drain was short, not just the first."""
    text = "One two three. Four five six. Seven eight nine. "
    pieces, remaining = segment(text, budget=100, first_budget=10, final=False)
    assert pieces == ["One two three.", "Four five six. Seven eight nine."]
    assert remaining == " "
    # The same text without the opening budget packs into one piece.
    assert segment(text, budget=100, final=False)[0] == [
        "One two three. Four five six. Seven eight nine."
    ]


def test_bc_39_opening_budget_bounds_unpunctuated_opening_text() -> None:
    """With nothing finished yet, streamed unpunctuated text is cut as soon as it passes the
    opening budget, not the full budget (C++ used 200 for the whole opening drain)."""
    buffer = ""
    pieces: list[str] = []
    while not pieces:
        buffer += "word "
        pieces, buffer = segment(buffer, budget=100, first_budget=20, final=False)
    assert pieces == ["word word word word word"]
    assert buffer == ""


def test_bc_38_budget_zero_means_no_length_splitting() -> None:
    """C++ turned split_chars 0 into 600 on the WebSocket; 0 now means no length splitting on
    both interfaces, and pieces are cut at sentence ends only."""
    text = "A long first sentence, with clauses. A second one! And an unfinished tail"
    assert split_text(text, budget=0) == [text]
    assert split_text(text, budget=0, first_budget=5) == [text]
    assert segment(text, budget=0, first_budget=5, final=False) == (
        ["A long first sentence, with clauses. A second one!"],
        " And an unfinished tail",
    )
    # Unpunctuated text is never cut by length.
    run = "中" * 500
    assert segment(run, budget=0, final=False) == ([], run)


def test_bc_39_cjk_without_punctuation_stays_bounded() -> None:
    """C++ drain only fell back to cutting at a space, so CJK with no punctuation never drained
    and the buffer grew without bound."""
    budget = 20
    pieces, remaining = segment("中" * 1000, budget=budget, final=False)
    assert weigh(remaining) <= budget
    assert all(weigh(p) <= budget for p in pieces)
    assert "".join(pieces) + remaining == "中" * 1000
    # Below 2 x budget an unbroken run waits for punctuation.
    assert segment("中" * 10, budget=budget, final=False) == ([], "中" * 10)


def test_bc_39_cjk_clause_mark_is_the_first_fallback() -> None:
    """Over budget with no sentence end, the cut is at the last clause mark or space."""
    text = "今天天气很好，我们去公园散步"
    assert segment(text, budget=20, final=False) == (["今天天气很好，"], "我们去公园散步")


def test_bc_39_weighted_length_everywhere() -> None:
    """C++ drain measured UTF-8 bytes (2 per Cyrillic letter) where split_text used weight (3);
    both now use weight."""
    assert weigh("aé中😀") == 1 + 3 + 3 + 3
    assert segment("привет как дела", budget=40, final=False) == (["привет как"], "дела")


# --- text details ---


def test_bc_44_tabs_and_carriage_returns_inside_a_piece_are_kept() -> None:
    """C++'s speaking.text dropped tabs and carriage returns; pieces now keep their exact inner
    text and are only stripped at the ends."""
    assert split_text("\t col1\tcol2\r\n", budget=100) == ["col1\tcol2"]
    assert segment("a\tb\rc. next", budget=100, final=False) == (["a\tb\rc."], " next")


def test_empty_and_whitespace_only_give_no_pieces() -> None:
    """C++ returned the empty or whitespace-only text as a piece. Pieces are now stripped and
    empty ones dropped (R11), which is why the goldens are normalized before comparing."""
    assert split_text("", budget=20) == []
    assert split_text("   \n   ", budget=20) == []
    assert segment("", budget=20, final=True) == ([], "")


# --- C++ behaviour the port keeps ---


@pytest.mark.parametrize(
    "prefix,run,suffix,budget",
    [
        ("short bits, more short bits, ", "x" * 50, ", and then a few more words after it.", 30),
        ("今天天气, ", "中" * 40, ", and then some more text.", 20),
    ],
    ids=["ascii_unsplittable_run", "cjk_unsplittable_run"],
)
def test_split_text_never_fragments_an_unsplittable_run(prefix, run, suffix, budget) -> None:
    """With the whole text known, a run with no clause mark or space stays in one piece (soft
    budget, as in C++); only the streaming buffer hard-cuts, to stay bounded."""
    pieces = split_text(prefix + run + suffix, budget=budget)
    assert len([p for p in pieces if run in p]) == 1


# --- property: the streaming buffer stays bounded and loses nothing ---

_ALPHABET = list("abcdefgh  ") + list(".,!?;:\"')\n\t3") + list("中文好。，、…．é")


def _no_space(text: str) -> str:
    return "".join(text.split())


def test_bc_39_streaming_leftover_stays_bounded() -> None:
    """C++ could hold an arbitrarily long unpunctuated buffer. Streamed in random chunks, a
    non-final leftover now weighs at most max(budget, first_budget) or 2 x budget, pieces are
    stripped and non-empty, and every non-space character comes out once, in order."""
    rng = random.Random(20260924)
    for _ in range(500):
        budget = rng.randint(1, 40)
        first_budget = rng.choice([0, rng.randint(1, 80)])
        bound = max(budget, first_budget, 2 * budget)
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 300)))

        spoken: list[str] = []
        buffer = ""
        pos = 0
        while pos < len(text):
            step = rng.randint(1, 25)
            buffer += text[pos : pos + step]
            pos += step
            # The session passes the opening budget until the first piece has been produced.
            opening = first_budget if not spoken else 0
            pieces, buffer = segment(buffer, budget=budget, first_budget=opening, final=False)
            spoken += pieces
            assert weigh(buffer) <= bound, (text, budget, first_budget, buffer)
        pieces, rest = segment(
            buffer, budget=budget, first_budget=first_budget if not spoken else 0, final=True
        )
        spoken += pieces

        assert rest == ""
        assert all(p and p == p.strip() for p in spoken)
        assert _no_space("".join(spoken)) == _no_space(text)
