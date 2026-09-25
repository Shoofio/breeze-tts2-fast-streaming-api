"""Split text into speakable pieces. One segmenter serves HTTP and the WebSocket drain.

Ported from Breeze-TTS-2.cpp's `src/text_split.cpp` (`split_text`, `split_sentences`,
`split_clauses`) and the `sentence_end`/`drain` helpers in `apps/server/ws_api.cpp`. The C++ had
two segmenters that disagreed; this module has one, with the fixes listed in
specs/003-cpp-compatible-api/research.md R11 (BC-39, BC-44, BC-46).

The shape is C++'s: text is cut into sentences, an over-budget sentence into clauses, and the
units are packed into pieces up to the budget. The budget is soft: a single unit heavier than it
stays whole. What differs from C++:

- Sentence ends: `\\n`; `.!?;` followed, after any closing quotes or brackets, by a space, tab,
  CR, LF or U+3000 (not a no-break or other typographic space: those keep `Dr.\\u00a0Smith`
  together); and the CJK stops `。！？；…．`, which absorb the stops and closers that follow
  them (`真的吗？！`, `好。」`). `．` between two digits is a decimal point (`３．１４`). In a
  non-final buffer a sentence end is only complete once the character after it has arrived, so
  `3.` then `14` stays `3.14`.
- Clauses: an over-budget sentence is cut after the first break once the clause weighs at least
  the budget. Breaks are whitespace other than LF (including no-break and typographic spaces),
  `，` and `、`. C++ also required the word before a space to be longer than a quarter of the
  budget, so ordinary prose was never cut. `,` and `:` are not breaks themselves; the space
  after them is, so `1,000`, `10:30` and `http://` stay whole.
- Unbroken runs: a clause that passes 2 x budget is closed at its last break; a run with no
  break at all is hard-cut once over 2 x budget, into chunks within budget, between grapheme
  clusters (see `_joins_previous`). So no unit weighs more than 2 x budget, give or take one
  cluster.
- Streaming: a non-final buffer gives up its complete sentences and every clause the rule above
  has closed; the rest waits. Weights are always weighted, never UTF-8 bytes.
- `first_budget` applies to the first returned piece only; `budget == 0` means no length limit.
- Pieces are stripped, and text with no letter or digit is dropped: TTS can't speak
  punctuation-only or emoji-only text. Tabs and carriage returns inside a piece are kept.

Pure functions, no I/O.
"""

from __future__ import annotations

import unicodedata
from itertools import accumulate, pairwise

_ASCII_STOPS = frozenset(".!?;")
# CJK punctuation carries its own spacing, so it ends a sentence without a following gap.
_CJK_STOPS = frozenset("。！？；…．")
_FULLWIDTH_PERIOD = "．"
# Closing quotes and brackets right after a stop belong to the sentence they close.
_CLOSERS = frozenset("\"')]}”’」』）》】〉〕〗〙〛］｝»›｣〞〟＂＇")
# What may follow `.!?;` (after closers) to end a sentence.
_GAPS = frozenset(" \t\r\n　")
# Word breaks inside a sentence: the gaps other than LF (which ends the sentence), typographic
# spaces that don't end a sentence, and the CJK clause marks.
_TYPOGRAPHIC_SPACES = frozenset(
    "   " + "".join(chr(c) for c in range(0x2000, 0x200B))
)
_CLAUSE_BREAKS = (_GAPS - {"\n"}) | _TYPOGRAPHIC_SPACES | frozenset("，、")
_ZWJ = "‍"
# The UAX #15 stream-safe limit: a cluster may be cut after this many joined code points, so a
# pathological run of combining marks can't make a unit, or the time spent on it, unbounded.
_MAX_JOINED = 30


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


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9" or "０" <= ch <= "９"


# --- grapheme clusters: a heuristic for the UAX #29 rules that matter here, no dependency ---


def _extends(ch: str) -> bool:
    """A character that always belongs to the one before it: a combining mark, ZWJ, variation
    selector, emoji skin-tone modifier or emoji tag."""
    cp = ord(ch)
    return (
        unicodedata.category(ch).startswith("M")
        or ch == _ZWJ
        or 0xFE00 <= cp <= 0xFE0F
        or 0xE0100 <= cp <= 0xE01EF
        or 0x1F3FB <= cp <= 0x1F3FF
        or 0xE0020 <= cp <= 0xE007F
    )


def _is_pictographic(ch: str) -> bool:
    """Rough Extended_Pictographic: the emoji and symbol blocks that ZWJ sequences use."""
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2190 <= cp <= 0x21FF
        or 0x2300 <= cp <= 0x23FF
        or 0x25A0 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D, 0x3297, 0x3299)
    )


def _is_regional(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _hangul_type(ch: str) -> str:
    """The Hangul syllable type of `ch`: L, V, T, LV, LVT, or '' for anything else."""
    cp = ord(ch)
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LV" if (cp - 0xAC00) % 28 == 0 else "LVT"
    return ""


def _joins_previous(prev: str, ch: str, before: str = "") -> bool:
    """Whether `ch` continues the grapheme cluster that `prev` is in, so no cut goes between.

    `before` is the character before `prev` when `prev` belongs to its cluster, else "".

    Nothing joins a break or a gap. Otherwise `ch` joins when it is a combining mark, ZWJ,
    variation selector, skin-tone modifier or tag; when it is a pictograph after a ZWJ; when it
    is a letter after a virama, or after a virama and a ZWJ (an Indic conjunct or half form;
    after any other ZWJ a letter starts a new cluster); or when it continues a Hangul syllable
    made of jamo. Flags (pairs of regional indicators) need the count of indicators before them,
    so `_cluster_starts` handles those.
    """
    if prev in _CLAUSE_BREAKS or prev in _GAPS:
        return False
    if _extends(ch):
        return True
    is_letter = unicodedata.category(ch) == "Lo"
    if prev == _ZWJ:
        return _is_pictographic(ch) or (is_letter and _is_virama(before))
    if _is_virama(prev):
        return is_letter
    before, after = _hangul_type(prev), _hangul_type(ch)
    return (
        (before == "L" and after in ("L", "V", "LV", "LVT"))
        or (before in ("LV", "V") and after in ("V", "T"))
        or (before in ("LVT", "T") and after == "T")
    )


def _is_virama(ch: str) -> bool:
    return ch != "" and unicodedata.combining(ch) == 9  # canonical combining class 9


def _cluster_starts(text: str) -> list[bool]:
    """For each index, whether a cut may go just before it. One linear pass.

    A cut is allowed where no cluster continues, or once a cluster has had `_MAX_JOINED` joined
    code points. The state resets at every allowed position, so the answer for an index depends
    only on the text since the last one; a buffer that starts at an earlier cut gets the same
    answers as the whole text.
    """
    allowed = [True] * (len(text) + 1)
    joined = 0
    regional = 1 if text and _is_regional(text[0]) else 0  # regional indicators in a row
    for c in range(1, len(text)):
        prev, ch = text[c - 1], text[c]
        # Look back past `prev` only inside the current cluster, so a buffer that starts at a
        # cut gets the same answer as the whole text.
        before = text[c - 2] if joined > 0 else ""
        joins = _joins_previous(prev, ch, before) or (
            _is_regional(ch) and regional % 2 == 1  # the second indicator of a flag
        )
        if joins and joined < _MAX_JOINED:
            allowed[c] = False
            joined += 1
            regional = regional + 1 if _is_regional(ch) else 0
        else:
            joined = 0
            regional = 1 if _is_regional(ch) else 0
    return allowed


# --- sentences, clauses, pieces ---


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
        elif ch in _CJK_STOPS and not (
            ch == _FULLWIDTH_PERIOD
            and 0 < i < n - 1
            and _is_digit(text[i - 1])
            and _is_digit(text[i + 1])
        ):
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


def _split_clauses(
    text: str, limit: int, hard_limit: int, first_limit: int = 0
) -> tuple[list[str], str]:
    """Cut `text` into clauses; return the closed clauses and the open tail.

    A clause closes after the first break once it weighs at least its limit (C++
    `split_clauses` without its quarter-budget condition). A clause that passes `hard_limit`
    because of a run with no break is closed at its last break instead; if the run alone is over
    `hard_limit`, it is hard-cut into chunks within the limit. The first clause or chunk uses
    `first_limit` when it is set, the rest `limit`.

    Each cut depends only on the text before it, so the same text gives the same clauses whether
    it arrives whole or in pieces. Linear in the length of `text`.
    """
    prefix = [0, *accumulate(_char_weight(ch) for ch in text)]
    allowed = _cluster_starts(text)
    closed: list[str] = []
    current = first_limit if first_limit > 0 else limit
    start = 0
    run_start = 0  # just after the last break, where the current unbroken run began

    def weight(end: int) -> int:
        return prefix[end] - prefix[start]

    def close(end: int) -> None:
        nonlocal start, current
        closed.append(text[start:end])
        start = end
        current = limit

    def chunk_end(stop: int) -> int | None:
        """The last allowed cut in (start, stop] within the limit, else the first one past it.
        At most _MAX_JOINED positions after that cut are scanned, which keeps this linear."""
        best = None
        for c in range(start + 1, stop + 1):
            over = weight(c) > current
            if over and best is not None:
                break
            if allowed[c]:
                best = c
                if over:
                    break
        return best

    for i, ch in enumerate(text):
        if ch in _CLAUSE_BREAKS:
            run_start = i + 1
            if weight(i + 1) >= current:
                close(i + 1)
        elif weight(i + 1) > hard_limit:
            if run_start > start:
                close(run_start)
            if weight(i + 1) > hard_limit:
                # Cut before characters already read (up to i), never inside a cluster.
                while weight(i + 1) > current:
                    end = chunk_end(i)
                    if end is None:
                        break  # one cluster so far: nowhere safe to cut yet
                    close(end)
                run_start = start
    return closed, text[start:]


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
    is explicit to give the same pieces either way. A unit with nothing to speak never starts a
    piece, so it is dropped unless it sits inside one; it doesn't use up the opening budget.
    """
    limit = first_budget if first_budget > 0 else budget
    out: list[str] = []
    cur = ""
    cw = 0
    closes = False
    for unit, unit_closes in units:
        w = weigh(unit)
        if cw > 0 and (closes or (budget > 0 and cw + w > limit)):
            out.append(cur.strip())
            cur = ""
            cw = 0
            limit = budget
        if not cur and not _speakable(unit):
            continue
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
    that are already closed (see `_split_clauses`); the open tail waits. When nothing before
    the rest is spoken now, its first clause is cut against the opening budget, so the first
    audio comes soon; later clauses use the budget. A non-final `remaining` weighs at most
    max(2 x budget, 3 x (_MAX_JOINED + 1)): the second term is one grapheme cluster with nowhere
    safe to cut yet.

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
        # The rest holds the opening piece only when nothing before it is spoken now.
        opening = 0 if _speakable(buffer[:cut]) or first_budget >= budget else first_budget
        closed, rest = _split_clauses(rest, budget, 2 * budget, opening)
        units += [(c, True) for c in closed]
    return _pack(units, budget, first_budget), rest


def split_text(text: str, *, budget: int, first_budget: int = 0) -> list[str]:
    """Split a whole text into pieces: `segment(text, ..., final=True)` without the remainder."""
    return segment(text, budget=budget, first_budget=first_budget, final=True)[0]
