"""Split text into speakable pieces. One segmenter serves HTTP and the WebSocket drain.

Ported from Breeze-TTS-2.cpp's `src/text_split.cpp` (`split_text`, `split_sentences`,
`split_clauses`) and the `sentence_end`/`drain` helpers in `apps/server/ws_api.cpp`. The C++ had
two segmenters that disagreed; this module has one, with the fixes listed in
specs/003-cpp-compatible-api/research.md R11 (BC-39, BC-44, BC-46).

The shape is C++'s: text is cut into sentences, an over-budget sentence into clauses, and the
units are packed into pieces up to the budget. The budget is soft: a single unit heavier than it
stays whole. What differs from C++:

- Sentence ends: `\\n`; `.!?;` followed, after any closing quotes or brackets, by a space, tab,
  CR, LF or U+3000 (not a no-break space, which is there to keep `Dr. Smith` together);
  and the CJK stops `。！？；…．`, which absorb the stops and closers that follow them
  (`真的吗？！`, `好。」`). In a non-final buffer a sentence end is only complete once the
  character after it has arrived, so `3.` then `14` stays `3.14`.
- Clauses: an over-budget sentence is cut after the first space, tab, CR, U+3000, `，` or `、`
  once the clause weighs at least the budget. C++ also required the word before a space to be
  longer than a quarter of the budget, so ordinary prose was never cut. `,` and `:` are not
  break points themselves; the space after them is, so `1,000`, `10:30` and `http://` stay whole.
- Unbroken runs: a clause that passes 2 x budget is closed at its last break; a run with no
  break at all is hard-cut once over 2 x budget, into chunks within budget, never inside a
  combining sequence. So no unit weighs more than 2 x budget.
- Streaming: a non-final buffer gives up its complete sentences and every clause the rule above
  has closed; the rest waits. Weights are always weighted, never UTF-8 bytes.
- `first_budget` applies to the first returned piece only; `budget == 0` means no length limit.
- Pieces are stripped, and text with no letter or digit is dropped: TTS can't speak
  punctuation-only or emoji-only text. Tabs and carriage returns inside a piece are kept.

Pure functions, no I/O.
"""

from __future__ import annotations

import unicodedata
from itertools import pairwise

_ASCII_STOPS = frozenset(".!?;")
# CJK punctuation carries its own spacing, so it ends a sentence without a following gap.
_CJK_STOPS = frozenset("。！？；…．")
# Closing quotes and brackets right after a stop belong to the sentence they close.
_CLOSERS = frozenset("\"')]}”’」』）》】〉〕〗〙〛］｝»›｣〞〟＂＇")
# What ends a clause, and (with LF) what may follow `.!?;` to end a sentence.
_SPACES = frozenset(" \t\r　")
_GAPS = _SPACES | {"\n"}
_CLAUSE_BREAKS = _SPACES | frozenset("，、")
_ZWJ = "‍"


def weigh(text: str) -> int:
    """Weighted length: ASCII counts 1, any other code point 3 (C++ text_split.cpp `weigh`).

    A non-ASCII character (a CJK syllable, an accented letter, an emoji) takes noticeably longer
    to speak than one ASCII letter.
    """
    return sum(_char_weight(ch) for ch in text)


def _char_weight(ch: str) -> int:
    return 1 if ord(ch) < 128 else 3


def _speakable(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in "LN" for ch in text)


def _joins_previous(ch: str) -> bool:
    """True for a character that belongs to the one before it: a combining mark, a zero-width
    joiner, a variation selector or an emoji skin-tone modifier."""
    return (
        unicodedata.category(ch).startswith("M")
        or ch == _ZWJ
        or "︀" <= ch <= "️"
        or "\U000e0100" <= ch <= "\U000e01ef"
        or "\U0001f3fb" <= ch <= "\U0001f3ff"
    )


def _sentence_ends(text: str, final: bool) -> list[int]:
    """Return the end index (exclusive) of every complete sentence in `text`.

    `.!?;` only ends a sentence when a gap follows it (after any closers), so `3.14` and `U.S.A`
    stay whole; `Dr. Smith` is cut, as in C++. A sentence end that reaches the end of a non-final
    buffer is not complete yet: the next character might be a digit or another closer.
    """
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in _ASCII_STOPS:
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if (j < n and text[j] in _GAPS) or (j == n and final):
                ends.append(j)
            i = j
        elif ch in _CJK_STOPS:
            j = i + 1
            while j < n and (text[j] in _CJK_STOPS or text[j] in _CLOSERS):
                j += 1
            if j < n or final:
                ends.append(j)
            i = j
        else:
            if ch == "\n" and (i + 1 < n or final):
                ends.append(i + 1)
            i += 1
    return ends


def _split_clauses(text: str, limit: int, hard_limit: int) -> tuple[list[str], str]:
    """Cut `text` into clauses; return the closed clauses and the open tail.

    A clause closes after the first break once it weighs at least `limit` (C++ `split_clauses`
    without its quarter-budget condition). A clause that passes `hard_limit` because of a run
    with no break is closed at its last break instead; if the run alone is over `hard_limit`, it
    is hard-cut into chunks within `limit`. Each cut depends only on the text before it, so the same text gives the same
    clauses whether it arrives whole or in pieces.
    """
    closed: list[str] = []
    start = 0
    run_start = 0  # just after the last break, where the current unbroken run began
    cw = 0
    run_w = 0
    for i, ch in enumerate(text):
        w = _char_weight(ch)
        cw += w
        run_w += w
        if ch in _CLAUSE_BREAKS:
            run_start = i + 1
            run_w = 0
            if cw >= limit:
                closed.append(text[start : i + 1])
                start = i + 1
                cw = 0
        elif cw > hard_limit:
            if run_start > start:
                closed.append(text[start:run_start])
                start = run_start
                cw = run_w
            if cw > hard_limit:
                chunks, start = _hard_cut(text, start, i + 1, limit)
                closed += chunks
                run_start = start
                cw = run_w = weigh(text[start : i + 1])
    return closed, text[start:]


def _hard_cut(text: str, start: int, stop: int, limit: int) -> tuple[list[str], int]:
    """Cut chunks off the front of the unbroken run `text[start:stop]` until what is left weighs
    at most `limit`. Each chunk is as long as fits within `limit` (at least one character), and
    a cut never separates a character from a combining mark, ZWJ or variation selector."""
    chunks: list[str] = []
    while weigh(text[start:stop]) > limit:
        cut = None
        w = 0
        for c in range(start + 1, stop):
            w += _char_weight(text[c - 1])
            if w > limit and cut is not None:
                break
            if not _joins_previous(text[c]) and text[c - 1] != _ZWJ:
                cut = c
                if w > limit:
                    break
        if cut is None:
            break  # one combining sequence: nowhere safe to cut yet
        chunks.append(text[start:cut])
        start = cut
    return chunks, start


def _units(text: str, ends: list[int], budget: int) -> list[tuple[str, bool]]:
    """Cut `text` into sentences at `ends` (plus any unfinished rest), then over-budget ones into
    clauses. With no length limit, sentences are the units.

    Each unit comes with a flag saying whether it is a closed clause (see `_pack`).
    """
    bounds = [0, *ends]
    if bounds[-1] < len(text):
        bounds.append(len(text))
    sentences = [text[a:b] for a, b in pairwise(bounds)]
    if budget <= 0:
        return [(s, False) for s in sentences]
    units: list[tuple[str, bool]] = []
    for sentence in sentences:
        closed, tail = _split_clauses(sentence, budget, 2 * budget)
        units += [(c, True) for c in closed]
        if tail:
            units.append((tail, False))
    return units


def _pack(units: list[tuple[str, bool]], budget: int, first_budget: int) -> list[str]:
    """Merge consecutive units into pieces up to the budget, then strip them.

    The first piece is packed against `first_budget` when it is set, every later one against
    `budget` (C++ `split_text`). The budget is soft: a single unit heavier than it stays whole.
    A piece always ends after a closed clause. In C++ that follows from the clause weighing at
    least the budget; here a clause can also close short (at a break before a long run, or as a
    hard-cut chunk), and a streamed clause is spoken before the next one arrives, so the rule
    is explicit to give the same pieces either way. Units with nothing to speak are dropped
    first, so they neither form a piece nor use up the opening budget.
    """
    limit = first_budget if first_budget > 0 else budget
    out: list[str] = []
    cur = ""
    cw = 0
    closes = False
    for unit, unit_closes in units:
        if not _speakable(unit):
            continue
        w = weigh(unit)
        if cw > 0 and (closes or (budget > 0 and cw + w > limit)):
            out.append(cur.strip())
            cur = ""
            cw = 0
            limit = budget
        cur += unit
        cw += w
        closes = unit_closes
    if cur:
        out.append(cur.strip())
    return out


def segment(
    buffer: str, *, budget: int, first_budget: int = 0, final: bool
) -> tuple[list[str], str]:
    """Take the pieces that are ready to speak out of `buffer`.

    Returns `(pieces, remaining)`. With `final`, everything is ready and `remaining` is empty.
    Otherwise the pieces are the complete sentences plus the clauses of the unfinished rest
    that are already closed (see `_split_clauses`); the open tail waits. For the opening piece
    of a session, that rest is cut against the opening budget, so the first audio comes soon.
    A non-final `remaining` weighs at most 2 x budget, give or take one combining sequence.

    `budget` is the weighted piece length; 0 means no length limit, so everything ready is one
    piece. `first_budget`, when positive, is the budget of the first returned piece only; the
    caller decides when a piece is the opening one.
    """
    if final:
        if budget <= 0 or weigh(buffer) <= budget:
            # The whole text is known and fits one piece. A lone piece anchors nothing, so it
            # needs no short opening piece either (C++ `split_text`).
            return _pack([(buffer, False)], 0, 0), ""
        units = _units(buffer, _sentence_ends(buffer, True), budget)
        return _pack(units, budget, first_budget), ""

    ends = _sentence_ends(buffer, False)
    cut = ends[-1] if ends else 0
    units = _units(buffer[:cut], ends, budget)
    rest = buffer[cut:]
    if budget > 0:
        # The rest becomes the opening piece only when nothing before it is spoken now.
        opening = first_budget if 0 < first_budget < budget else budget
        limit = budget if _speakable(buffer[:cut]) else opening
        closed, rest = _split_clauses(rest, limit, 2 * budget)
        units += [(c, True) for c in closed]
    return _pack(units, budget, first_budget), rest


def split_text(text: str, *, budget: int, first_budget: int = 0) -> list[str]:
    """Split a whole text into pieces: `segment(text, ..., final=True)` without the remainder."""
    return segment(text, budget=budget, first_budget=first_budget, final=True)[0]
