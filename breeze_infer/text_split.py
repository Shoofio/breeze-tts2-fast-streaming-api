"""Split text into speakable pieces. One segmenter serves HTTP and the WebSocket drain.

Ported from Breeze-TTS-2.cpp's `src/text_split.cpp` (`split_text`, `split_sentences`,
`split_clauses`) and the `sentence_end`/`drain` helpers in `apps/server/ws_api.cpp`. The C++ had
two segmenters that disagreed; this module has one, with the fixes listed in
specs/003-cpp-compatible-api/research.md R11 (BC-39, BC-44, BC-46):

- One stop set on both interfaces: `\\n`; `.!?;` followed (after any closing `"')]`) by
  whitespace or the end of a final buffer; and the CJK stops `。！？；…．`.
- Closing quotes and brackets after `.!?;` stay with their sentence on both interfaces.
- A sentence end at the very end of a non-final buffer waits for the next character, so `3.`
  followed later by `14` stays whole.
- Lengths are always weighted (see `weigh`), never UTF-8 bytes.
- Unpunctuated non-final text is cut at the last clause mark or space once it is over budget, and
  hard-cut once an unbroken run passes 2 x budget, so the streaming buffer stays bounded.
- `first_budget` applies to the first returned piece only.
- `budget == 0` means no length splitting.
- Pieces are stripped and empty ones dropped; tabs and carriage returns inside a piece are kept.

Pure functions, no I/O. Python strings are indexed by code point, so there is no UTF-8 byte
bookkeeping here.
"""

from __future__ import annotations

from itertools import pairwise

_ASCII_STOPS = frozenset(".!?;")
# CJK punctuation carries its own spacing, so it ends a sentence without a following gap.
_CJK_STOPS = frozenset("。！？；…．")
# Closing quotes and brackets right after `.!?;` belong to the sentence they close.
_CLOSERS = frozenset("\"')]")
# Where an over-budget sentence may be broken into clauses.
_CLAUSE_MARKS = frozenset(",，、:")


def weigh(text: str) -> int:
    """Weighted length: ASCII counts 1, any other code point 3 (C++ text_split.cpp `weigh`).

    A non-ASCII character (a CJK syllable, an accented letter, an emoji) takes noticeably longer
    to speak than one ASCII letter.
    """
    return sum(_char_weight(ch) for ch in text)


def _char_weight(ch: str) -> int:
    return 1 if ord(ch) < 128 else 3


def _sentence_ends(text: str, final: bool) -> list[int]:
    """Return the end index (exclusive) of every complete sentence in `text`.

    `.!?;` only ends a sentence when whitespace follows it (after any closing quotes), so `3.14`
    and `U.S.A` stay whole; `Dr. Smith` is cut, as in C++. In a non-final buffer, a sentence end
    that reaches the end of the buffer is not complete yet: the next character might be a digit
    or another closing quote.
    """
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        i += 1
        if ch in _ASCII_STOPS:
            j = i
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j < n and not text[j].isspace():
                continue
            i = j
        elif ch != "\n" and ch not in _CJK_STOPS:
            continue
        if i == n and not final:
            break
        ends.append(i)
    return ends


def _split_clauses(sentence: str, budget: int) -> list[str]:
    """Break an over-budget sentence at clause marks, or at spaces that are not too close together.

    `since_break` is the weight since the last clause mark or space, so a space only counts as a
    cut point when the word before it clears a quarter of the budget. A run with no clause mark or
    space stays whole (C++ `split_clauses`).
    """
    if weigh(sentence) <= budget:
        return [sentence]
    out: list[str] = []
    start = 0
    cw = 0
    since_break = 0
    for i, ch in enumerate(sentence):
        w = _char_weight(ch)
        cw += w
        since_break += w
        comma = ch in _CLAUSE_MARKS
        space = ch == " "
        if cw >= budget and (comma or (space and since_break > budget // 4)):
            out.append(sentence[start : i + 1])
            start = i + 1
            cw = 0
            since_break = 0
        elif comma or space:
            since_break = 0
    if start < len(sentence):
        out.append(sentence[start:])
    return out


def _units(text: str, ends: list[int], budget: int) -> list[str]:
    """Cut `text` into sentences at `ends` (plus any unfinished rest), then over-budget ones into
    clauses. With no length splitting, sentences are the units."""
    bounds = [0, *ends]
    if bounds[-1] < len(text):
        bounds.append(len(text))
    sentences = [text[a:b] for a, b in pairwise(bounds)]
    if budget <= 0:
        return sentences
    return [clause for s in sentences for clause in _split_clauses(s, budget)]


def _last_break(text: str) -> int:
    """Index of the last clause mark or space in `text`, or -1."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == " " or text[i] in _CLAUSE_MARKS:
            return i
    return -1


def _hard_cut(run: str, budget: int) -> tuple[list[str], str]:
    """Cut chunks of at most `budget` weight (at least one character each) off the front of an
    unbroken run until what is left weighs at most `budget`."""
    chunks: list[str] = []
    rest_weight = weigh(run)
    start = 0
    while rest_weight > budget:
        end = start
        w = 0
        while end < len(run) and (end == start or w + _char_weight(run[end]) <= budget):
            w += _char_weight(run[end])
            end += 1
        chunks.append(run[start:end])
        rest_weight -= w
        start = end
    return chunks, run[start:]


def _pack(units: list[str], budget: int, first_budget: int) -> list[str]:
    """Merge consecutive units into pieces up to the budget, then strip them and drop empties.

    The first piece is packed against `first_budget` when it is set, every later one against
    `budget` (C++ `split_text`). The budget is soft: a single unit heavier than it stays whole.
    A whitespace-only accumulation never closes a piece, so the opening budget really lands on
    the first piece that is returned.
    """
    no_limit = budget <= 0
    limit = first_budget if first_budget > 0 else budget
    out: list[str] = []
    cur = ""
    cw = 0
    for unit in units:
        w = weigh(unit)
        if not no_limit and cur.strip() and cw + w > limit:
            out.append(cur)
            cur = ""
            cw = 0
            limit = budget
        cur += unit
        cw += w
    out.append(cur)
    return [p for p in (piece.strip() for piece in out) if p]


def segment(
    buffer: str, *, budget: int, first_budget: int = 0, final: bool
) -> tuple[list[str], str]:
    """Take the pieces that are ready to speak out of `buffer`.

    Returns `(pieces, remaining)`. With `final`, everything is ready and `remaining` is empty.
    Otherwise the buffer is cut after its last complete sentence, and the unfinished rest is kept,
    unless it is over budget:
    - it is cut after its last clause mark or space;
    - an unbroken run left after that (CJK without punctuation) is hard-cut once it weighs more
      than 2 x budget.
    So a non-final `remaining` weighs at most max(budget, first_budget) or 2 x budget.

    `budget` is the weighted piece length, and 0 means no length splitting (pieces are cut at
    sentence ends only). `first_budget`, when positive, is the budget of the first returned
    piece only; the caller decides when a piece is the opening one.
    """
    if final:
        # The whole text is known, and a lone piece needs no short opening piece to anchor
        # later ones, so text within budget stays one piece (C++ `split_text`).
        if budget <= 0 or weigh(buffer) <= budget:
            return _pack([buffer], 0, 0), ""
        units = _units(buffer, _sentence_ends(buffer, True), budget)
        return _pack(units, budget, first_budget), ""

    ends = _sentence_ends(buffer, False)
    cut = ends[-1] if ends else 0
    units = _units(buffer[:cut], ends, budget)
    rest = buffer[cut:]

    if budget > 0:
        opening = first_budget if first_budget > 0 else budget
        # The rest becomes the first piece only when nothing before it is spoken now.
        rest_limit = budget if buffer[:cut].strip() else opening
        if weigh(rest) > rest_limit:
            brk = _last_break(rest)
            if brk >= 0:
                units += _split_clauses(rest[: brk + 1], budget)
                rest = rest[brk + 1 :]
            if weigh(rest) > 2 * budget:
                chunks, rest = _hard_cut(rest, budget)
                units += chunks

    return _pack(units, budget, first_budget), rest


def split_text(text: str, *, budget: int, first_budget: int = 0) -> list[str]:
    """Split a whole text into pieces: `segment(text, ..., final=True)` without the remainder."""
    return segment(text, budget=budget, first_budget=first_budget, final=True)[0]
