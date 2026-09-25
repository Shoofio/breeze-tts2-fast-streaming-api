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
import time
import unicodedata
from pathlib import Path

import pytest

from breeze_infer.text_split import (
    _MAX_JOINED,
    _joins_previous,
    segment,
    split_text,
    weigh,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "cpp_golden" / "golden.json").read_text(encoding="utf-8")
)
SPLIT_CASES = {c["name"]: c for c in GOLDEN["split_text"]}
DRAIN_CASES = {c["name"]: c for c in GOLDEN["drain"]}

# Case name -> (expected new output, BC id). A split case's output is its piece list; a drain
# case's output is (pieces, remaining).
INTENTIONAL_DIFFERENCES: dict[str, tuple[object, str]] = {
    # R11: C++ treats a NUL after `.` as a closing quote (strchr matches the terminator), so
    # `a.\0` ended a sentence. NUL is rejected at validation now; here it is ordinary text.
    "nul_byte_in_text": (["a.\x00 b", "c d e", "f g h", "i j k", "l m n"], "BC-46"),
    # Finding 1: C++ split_clauses only cut at a space after a word longer than a quarter of
    # the budget, so ordinary prose ran far over budget. Now the first break once the clause
    # reaches the budget cuts.
    "english_multi_sentence": (
        ["The quick brown fox jumps over", "the lazy dog.", "It was a sunny day!",
         "Are you coming? Yes."],
        "BC-39",
    ),
    "sentence_longer_than_budget_clause_split": (
        ["This is a single very long sentence with", "no terminal punctuation yet, containing",
         "several commas, which should force clause", "splitting, because it exceeds the budget",
         "by quite a lot in total weight"],
        "BC-39",
    ),
    "long_run_no_punctuation_space_split": (
        ["supercalifragilisticexpialidocious", "word another word yetanotherword",
         "andmore words continuing on and", "on without any commas or periods",
         "anywhere in this run at all"],
        "BC-39",
    ),
    "first_budget_smaller_than_budget": (
        ["The quick brown fox jumps over", "the lazy dog.", "It was a sunny day!",
         "Are you coming? Yes indeed."],
        "BC-39",
    ),
    # Finding 6: emoji-only pieces are dropped; TTS can't speak them.
    "emoji_and_accented_multibyte": (["ab, 😀😀", "cd, éé", "ff, 😀😀😀😀", "gg."], "BC-39"),
    # Finding 2: C++ drain cut the unfinished rest at the buffer's last space, however long the
    # piece; the rest now gives up only the clauses the budget has closed.
    "drain_no_sentence_end_over_budget_space_split": (
        (["this buffer has no terminal punctuation", "at all but it is definitely longer"],
         "than the budget so"),
        "BC-39",
    ),
    "drain_ellipsis_not_recognized_by_sentence_end": (
        (["Wait for it…", "more words keep coming without", "any real stop here at all so it"],
         "runs long"),
        "BC-39",
    ),
    "drain_byte_vs_weight_asymmetry_drained": ((["привет как дела"], "сегодня хорошо"), "BC-39"),
    # R11: C++ drain compared UTF-8 bytes (56 <= 60) and kept the buffer; by weight it is 82.
    "drain_byte_vs_weight_asymmetry_undrained": (
        (["привет как дела сегодня"], "хорошо"),
        "BC-39",
    ),
    # R11: C++ drain didn't absorb the closing quote, so `."` never ended a sentence.
    "drain_no_quote_absorption": ((['He said "done."'], " next"), "BC-39"),
    # R11: C++ drain's stop set left out the fullwidth period that split_text had.
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
    """C++ let clauses run far over budget (split_clauses skipped spaces after short words),
    drained the unfinished rest at the buffer's last space however long the piece, spoke
    emoji-only pieces, drained with its own stop set and measured the buffer in UTF-8 bytes."""
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


def test_bc_39_only_plain_whitespace_ends_a_sentence() -> None:
    """C++ accepted only ' ' and '\\n' after `.!?;`. Tab, CR and the ideographic space (U+3000)
    now end a sentence too; a no-break space (U+00A0) keeps `Dr.\\u00a0Smith` together, which
    is what it is for."""
    assert segment("Dr. Smith and", budget=100, final=False) == ([], "Dr. Smith and")
    assert split_text("Dr. Smith. Yes.", budget=10) == ["Dr. Smith.", "Yes."]
    for gap in ("\t", "\r", "　"):
        assert segment(f"Hi.{gap}there", budget=100, final=False) == (["Hi."], f"{gap}there")


def test_bc_39_fullwidth_period_between_digits_is_a_decimal_point() -> None:
    """`．` is a CJK stop, but between two digits it is a decimal point: `３．１４`."""
    for number in ("３．１４", "3．14", "３．14"):
        text = f"圆周率是{number}吗"
        assert segment(text, budget=100, final=False) == ([], text)
        assert segment(text + "。好", budget=100, final=False) == ([text + "。"], "好")
    assert segment("好．然后", budget=100, final=False) == (["好．"], "然后")


def test_bc_39_exotic_spaces_break_clauses_but_do_not_end_sentences() -> None:
    """No-break, narrow no-break, fixed-width and medium mathematical spaces are word breaks,
    so long text using them still gets cut, but they don't end a sentence after `.`."""
    for gap in ("\u00a0", "\u202f", "\u2002", "\u2009", "\u200a", "\u205f"):
        assert split_text(f"aaaaaaaa{gap}bbbb", budget=9) == ["aaaaaaaa", "bbbb"]
        assert segment(f"Dr.{gap}Smith and", budget=100, final=False) == (
            [],
            f"Dr.{gap}Smith and",
        )


def test_bc_39_newline_cuts_on_the_websocket_too() -> None:
    """C++ drain's sentence_end ignored `\\n`, so line-separated text waited for the budget."""
    assert segment("first line\nsecond", budget=30, final=False) == (["first line"], "second")
    assert split_text("first line\nsecond line", budget=15) == ["first line", "second line"]


# --- sentence ends, closers and clause marks ---


def test_bc_39_cjk_stop_absorbs_following_stops_and_closers() -> None:
    """C++ cut after every CJK stop, so `真的吗？！` left a piece that was only `！`."""
    assert split_text("真的吗？！好的。", budget=20) == ["真的吗？！", "好的。"]
    assert segment("好。」你", budget=20, final=False) == (["好。」"], "你")
    # The run of stops might continue, so it waits at the end of a non-final buffer.
    assert segment("好！！！", budget=20, final=False) == ([], "好！！！")
    assert split_text("好！！！", budget=20) == ["好！！！"]


def test_bc_39_typographic_and_cjk_closers_are_absorbed() -> None:
    """C++ only absorbed `"')]`, so a curly or CJK closing quote after a stop blocked the cut
    (after `.`) or started the next piece (after `。`)."""
    assert segment("She said “yes.” Then", budget=30, final=False) == (
        ["She said “yes.”"],
        " Then",
    )
    assert segment("他说：“好。”然后", budget=30, final=False) == (["他说：“好。”"], "然后")
    assert segment("『好。』你", budget=30, final=False) == (["『好。』"], "你")


def test_bc_39_ascii_clause_marks_need_following_whitespace() -> None:
    """C++ broke at any `,` or `:`, so `1,000`, `10:30` and `http://` could be split. Now the
    space after them is the break."""
    assert split_text("aaaaaaaa 1,000,000 x", budget=9) == ["aaaaaaaa", "1,000,000", "x"]
    assert split_text("aaaaaaaa 10:30:00 x", budget=9) == ["aaaaaaaa", "10:30:00", "x"]
    assert split_text("aaaaaaaa http://x.io x", budget=9) == ["aaaaaaaa", "http://x.io", "x"]
    assert split_text("aaaaaaaa, bb", budget=9) == ["aaaaaaaa,", "bb"]
    # A comma at the end of a non-final buffer waits to see what follows.
    assert segment("abcdefgh,", budget=5, final=False) == ([], "abcdefgh,")
    assert segment("abcdefgh, ij", budget=5, final=False) == (["abcdefgh,"], "ij")


# --- budgets ---


def test_bc_39_spaced_text_without_punctuation_is_cut_before_the_budget() -> None:
    """C++ split_clauses only cut at a space after a word longer than a quarter of the budget,
    so ordinary unpunctuated prose was never cut: `word ` x 400 at 600 was one 2,000-weight
    piece, too long for the model's context."""
    words = " ".join(["word"] * 120)
    assert split_text("word " * 400, budget=600) == [words, words, words, " ".join(["word"] * 40)]
    assert split_text("The quick brown fox jumps over the lazy dog.", budget=30) == [
        "The quick brown fox jumps over",
        "the lazy dog.",
    ]


def test_bc_39_opening_budget_applies_to_the_first_piece_only() -> None:
    """C++ drained the whole opening buffer against the 200 budget, so every piece of that
    drain was short, not just the first."""
    text = "One two. Three four. Five six. "
    pieces, remaining = segment(text, budget=100, first_budget=10, final=False)
    assert pieces == ["One two.", "Three four. Five six."]
    assert remaining == " "
    # The same text without the opening budget packs into one piece.
    assert segment(text, budget=100, final=False)[0] == ["One two. Three four. Five six."]


def test_bc_39_opening_budget_bounds_unpunctuated_opening_text() -> None:
    """With nothing finished yet, streamed unpunctuated text is cut as soon as it passes the
    opening budget, not the full budget (C++ used 200 for the whole opening drain)."""
    buffer = ""
    pieces: list[str] = []
    while not pieces:
        buffer += "word "
        pieces, buffer = segment(buffer, budget=100, first_budget=20, final=False)
    assert pieces == ["word word word word"]
    assert buffer == ""


def test_bc_39_opening_budget_applies_to_the_first_streamed_clause_only() -> None:
    """The opening budget cut every clause of the opening drain short, as C++'s 200 budget did
    for the whole drain; it now shapes only the first piece."""
    pieces, rest = segment("word " * 60, budget=100, first_budget=20, final=False)
    twenty = " ".join(["word"] * 20)
    assert pieces == ["word word word word", twenty, twenty]
    assert rest == "word " * 16

    text = "中" * 1000
    pieces, rest = segment(text, budget=600, first_budget=200, final=False)
    assert 150 < weigh(pieces[0]) <= 200
    assert [weigh(p) for p in pieces[1:]] == [600] * (len(pieces) - 1)
    assert len(pieces) > 1
    assert "".join(pieces) + rest == text


def test_bc_38_budget_zero_means_no_length_limit() -> None:
    """C++ turned split_chars 0 into 600 on the WebSocket; 0 now means no length limit on both
    interfaces, so all the text that is ready is one piece."""
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
    assert weigh(remaining) <= 2 * budget
    assert all(weigh(p) <= budget for p in pieces)
    assert "".join(pieces) + remaining == "中" * 1000
    # Up to 2 x budget an unbroken run waits for punctuation.
    assert segment("中" * 13, budget=budget, final=False) == ([], "中" * 13)
    # The whole text is hard-cut the same way, so HTTP and WebSocket agree.
    assert split_text("中" * 1000, budget=budget)[:-1] == pieces


def test_bc_39_cjk_clause_mark_is_the_first_fallback() -> None:
    """Over budget with no sentence end, the cut is at the last clause mark or space."""
    text = "今天天气很好，我们去公园散步"
    assert segment(text, budget=20, final=False) == (["今天天气很好，"], "我们去公园散步")


def test_bc_39_weighted_length_everywhere() -> None:
    """C++ drain measured UTF-8 bytes (2 per Cyrillic letter) where split_text used weight (3);
    both now use weight."""
    assert weigh("aé中😀") == 1 + 3 + 3 + 3
    # 'привет как дела ' weighs 42 but is only 29 UTF-8 bytes, so C++ drain wouldn't cut it.
    assert segment("привет как дела сегодня", budget=40, final=False) == (
        ["привет как дела"],
        "сегодня",
    )


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


def test_bc_39_pieces_without_a_letter_or_digit_are_dropped() -> None:
    """C++ sent punctuation-only and emoji-only pieces to the model, which can't speak them."""
    assert split_text("。", budget=20) == []
    assert split_text("😀😀😀", budget=20) == []
    assert split_text("Hi 😀", budget=20) == ["Hi 😀"]


def _regional(ch: str) -> bool:
    return "\U0001f1e6" <= ch <= "\U0001f1ff"


def _assert_cuts_keep_clusters(text: str, pieces: list[str]) -> None:
    """Every piece boundary falls between two grapheme clusters: nothing joins across it and a
    flag's two regional indicators stay together."""
    pos = 0
    for piece in pieces:
        start = text.index(piece, pos)
        pos = start + len(piece)
        for cut in (start, pos):
            if 0 < cut < len(text):
                before = text[cut - 2] if cut >= 2 else ""
                assert not _joins_previous(text[cut - 1], text[cut], before), (text, piece)
                regional_before = 0
                while cut - regional_before > 0 and _regional(text[cut - regional_before - 1]):
                    regional_before += 1
                assert regional_before % 2 == 0, (text, piece)


@pytest.mark.parametrize(
    "run",
    [
        "e\u0301" * 40,
        "a👨\u200d👩\u200d👧" * 10,
        "中\ufe0f" * 30,
        "\u1100\u1161\u11a8" * 20,
        "a🇺🇸🇯🇵" * 15,
        "क्ष" * 30,
    ],
    ids=["combining_accent", "zwj_family", "variation_selector", "hangul_jamo", "flags", "virama"],
)
def test_bc_39_hard_cut_keeps_grapheme_clusters_whole(run) -> None:
    """A hard cut never separates a character from its combining mark, ZWJ or variation
    selector, a Hangul syllable's jamo, a flag's two regional indicators, or a virama from
    the consonant it joins."""
    for pieces in (split_text(run, budget=5), segment(run, budget=5, final=False)[0]):
        assert len(pieces) > 1
        _assert_cuts_keep_clusters(run, pieces)


def test_bc_39_zwj_joins_only_a_pictograph() -> None:
    """After a ZWJ only an emoji continues the cluster; a letter starts a new one."""
    assert _joins_previous("\u200d", "👩")
    assert not _joins_previous("\u200d", "b")
    assert not _joins_previous(" ", "\u0301")
    # A ZWJ after a virama makes a half form: the consonant after it stays in the cluster.
    assert _joins_previous("\u200d", "ष", "\u094d")


def test_bc_39_half_form_after_virama_and_zwj_stays_whole() -> None:
    """`क्‍ष` (virama + ZWJ + consonant) could be cut after the ZWJ."""
    unit = "क\u094d\u200dष"
    for pieces in (split_text(unit * 20, budget=4), segment(unit * 20, budget=4, final=False)[0]):
        assert len(pieces) > 1
        assert all(p == unit * (len(p) // len(unit)) for p in pieces), pieces


@pytest.mark.parametrize(
    "text",
    [
        "a" + "\u0301" * 10_000,
        "a\u200d" * 5_000,
        "\u1100" * 10_000,
        "🇺🇸" * 5_000,
        "a" + "\ufe0f" * 10_000,
    ],
    ids=["combining_marks", "zwj_letters", "hangul_leading_jamo", "flags", "variation_selectors"],
)
def test_bc_39_long_joined_runs_are_cut_in_linear_time(text) -> None:
    """A run of 10,000 combining marks took 12-14 s (quadratic). A joined run is now force-cut
    after 30 joined code points, the UAX #15 stream-safe limit, and scanning is linear."""
    began = time.perf_counter()
    whole = split_text(text, budget=20)
    streamed = segment(text, budget=20, final=False)
    assert time.perf_counter() - began < 0.5
    bound = max(2 * 20, 3 * (_MAX_JOINED + 1))
    assert all(weigh(piece) <= bound for piece in whole + streamed[0])
    assert weigh(streamed[1]) <= bound


# --- behaviour the port keeps ---


@pytest.mark.parametrize(
    "prefix,run,suffix,budget",
    [
        ("short bits, more short bits, ", "x" * 50, ", and then a few more words after it.", 30),
        ("今天天气, ", "中" * 13, ", and then some more text.", 20),
    ],
    ids=["ascii_unbroken_run", "cjk_unbroken_run"],
)
def test_an_unbroken_run_within_twice_the_budget_stays_whole(prefix, run, suffix, budget) -> None:
    """A run with no clause mark or space is kept whole up to 2 x budget (soft budget, as in
    C++); the clause before it is closed at its last break so the run fits."""
    pieces = split_text(prefix + run + suffix, budget=budget)
    assert len([p for p in pieces if run in p]) == 1


# --- properties ---

# Tokens, not single characters, so combining sequences stay well formed.
_TOKENS = (
    list("abcdefgh3") + ["  ", " ", " ", "\t", " ", "\n"]
    + list(".,!?;:\"')") + ["1,000", "10:30", "http://x.io"]
    + list("中文好。，、…．！？") + ["é", "é", "👨‍👩‍👧", "❤️"]
    + list("”’」』）》】〉")
    + ["\u202f", "\u2009", "한", "\u1100\u1161", "🇺🇸", "क्ष", "３．１４"]
)
# Tokens without a sentence stop, for the chunking property.
_UNSTOPPED_TOKENS = [t for t in _TOKENS if not any(ch in t for ch in "\n.!?;。！？…．")] + [
    "x.io"
]
# The heaviest possible grapheme cluster: a base and _MAX_JOINED joined code points, all
# non-ASCII. The open tail can be one such cluster with nowhere safe to cut.
_MAX_CLUSTER_WEIGHT = 3 * (_MAX_JOINED + 1)


def _random_text(rng: random.Random, tokens: list[str], max_tokens: int = 200) -> str:
    return "".join(rng.choice(tokens) for _ in range(rng.randint(0, max_tokens)))


def _stream(
    text: str, rng: random.Random, budget: int, first_budget: int, bound: int | None = None
) -> list[str]:
    """Feed `text` to segment in random chunks the way the WebSocket session does, then flush."""
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
        if bound is not None:
            assert weigh(buffer) <= bound, (text, budget, first_budget, buffer)
    opening = first_budget if not spoken else 0
    pieces, rest = segment(buffer, budget=budget, first_budget=opening, final=True)
    assert rest == ""
    return spoken + pieces


def _speakable(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def test_bc_39_streaming_leftover_stays_bounded() -> None:
    """C++ could hold an arbitrarily long unpunctuated buffer. Streamed in random chunks, a
    non-final leftover now weighs at most max(2 x budget, one grapheme cluster), whatever the
    opening budget; pieces are stripped, speakable and never cut inside a grapheme cluster; and
    every letter and digit comes out once, in order."""
    rng = random.Random(20260924)
    for _ in range(500):
        budget = rng.randint(1, 40)
        first_budget = rng.choice([0, rng.randint(1, 80)])
        bound = max(2 * budget, _MAX_CLUSTER_WEIGHT)
        text = _random_text(rng, _TOKENS)
        spoken = _stream(text, rng, budget, first_budget, bound)

        assert all(p == p.strip() and _speakable(p) for p in spoken)
        _assert_cuts_keep_clusters(text, spoken)
        assert _speakable("".join(spoken)) == _speakable(text)


def _within_soft_budget(piece: str, budget: int) -> bool:
    """The soft-budget rule: a piece over budget is a single clause that was cut at the first
    break after reaching the budget, so everything before its last inner break is under it."""
    if weigh(piece) <= budget:
        return True
    inner = [i for i, ch in enumerate(piece[:-1]) if ch in " \t\r\u3000，、"]
    return not inner or weigh(piece[: inner[-1]]) < budget


def test_bc_39_pieces_stay_within_budget_when_breaks_exist() -> None:
    """C++ let a clause of short words run on past the budget. When every word is shorter than
    the budget, a piece is now over it only as a single clause that overshoots by the word in
    progress, and the first piece only as a single unit (the opening budget is soft), whether
    the text arrives whole or in random chunks."""
    words = ["a", "to", "the", "word", "speech", "中", "中文"]
    separators = [" ", " ", ", ", ". ", "! ", "，", "。", "\n", "” "]
    rng = random.Random(7)
    for _ in range(300):
        budget = rng.randint(12, 60)
        first_budget = rng.choice([0, rng.randint(1, 80)])
        text = "".join(
            rng.choice(words) + rng.choice(separators) for _ in range(rng.randint(0, 80))
        )
        for pieces in (
            split_text(text, budget=budget, first_budget=first_budget),
            _stream(text, rng, budget, first_budget),
        ):
            for piece in pieces:
                assert _within_soft_budget(piece, max(first_budget, budget)), (text, piece)
            for piece in pieces[1:]:
                assert _within_soft_budget(piece, budget), (text, budget, piece)


def test_bc_39_chunking_does_not_change_the_pieces() -> None:
    """C++'s drain cut wherever the buffer happened to end, so the pieces depended on how the
    client chunked its messages. Text with no sentence end now gives the same pieces whole or
    in any chunks. Two limits: with sentence ends it can't, because the WebSocket speaks each
    finished sentence at once where the whole text would pack it with the next; and without
    the opening budget, because the WebSocket cuts an unfinished opening clause against it for
    a quick first audio, where HTTP keeps the first clause whole (soft budget, US3)."""
    rng = random.Random(11)
    for _ in range(500):
        budget = rng.randint(1, 40)
        text = _random_text(rng, _UNSTOPPED_TOKENS)
        whole = split_text(text, budget=budget)
        assert _stream(text, rng, budget, 0) == whole, (text, budget)
