"""`POST /v1/audio/speech` field parsing (data-model.md `SpeechRequest`, `ReferenceSpec`).

tasks.md T037 built the happy path: defaults, BC-02 (empty means absent), BC-09 (blank
instruction means the default), BC-10 (missing/blank text is `400 text_required`) and
FR-006's `0` -> `None` for the sampling fields. T048 (contracts/http-api.md, data-model.md
`SpeechRequest`) completes it: the strict ASCII-only number grammar with ranges (T048's
`_INT_LITERAL`/`_DECIMAL_LITERAL` below), duplicate-field detection (BC-08,
`_check_no_duplicate_names`), length limits and the control-character rule (BC-05, BC-46),
and the full `reference_conflict` / `ref_text_required` / `reference_required` ordering
(`_build_reference`). `read_fields` runs its query-only checks -- a duplicate wholly within
the query string, and `ref_audio` given as a query parameter (`_reject_ref_audio_text`) --
before the request's content type is even sniffed, so a rejected content type, an
unsupported charset, or a large multipart parse can never surface their own error ahead of
one already knowable from the query string alone (T048 post-final review findings #4/#7).
The multipart and urlencoded branches each then run their own combined duplicate check (the
query's names plus that branch's) and their own ref_audio-as-text check -- multipart's is
`_split_multipart_fields`'s own file-vs-text check, done in wire order, rather than a
separate call -- before any other per-field check ever runs; the no-body branch has nothing
left to check a second time. `Fields` itself
keeps the form and the query string as separate multi-dicts so `_first` can read "the one
value" from either source without needing to merge them first -- by the time `Fields` is
built, `_check_no_duplicate_names` has already ruled out either one holding more than one
value for the same name.

research.md R1: no `Form(...)` parameters -- FastAPI's silently keeps the last of a
duplicate field, and its parser limits can't be changed. research.md R6: a truncated
multipart body parses as an empty form with status 200, so `text` (and every other
required field) has to be checked explicitly here rather than relying on the parser to
reject a short body.

**Two review passes since** found `request.form()` itself -- Starlette's own parser, not
FastAPI's `Form()` -- and `request.query_params` too permissive for this contract, in ways
plain happy-path testing didn't exercise. `read_fields` below never calls either; instead it
drives everything by hand, off `request.scope["query_string"]` and the body:

- `application/x-www-form-urlencoded` (and the query string, which is the same wire format)
  goes through `_urlencoded_pairs_sync`, built on `urllib.parse.parse_qsl` -- but *not* using
  its `encoding`/`errors` parameters, which are silently ignored when its input is `bytes`
  (verified against cpython's `urllib/parse.py`: the bytes branch's `_unquote` calls
  `unquote_to_bytes` directly, with no decode step at all). `parse_qsl` is used purely for
  its percent-decoding and its `max_num_fields` guard (a `ValueError` before any per-field
  work at all, review 2 findings #1/#2/#9); the actual UTF-8 decode, `errors="strict"`, is
  done here, by hand, on the bytes it returns -- which is also why a raw, un-percent-escaped
  UTF-8 sequence in the body decodes correctly too, not just a `%XX`-escaped one:
  `unquote_to_bytes` passes bytes it doesn't recognize as an escape straight through, so
  either spelling reaches this module's own decode step as the same bytes. A body over
  `_TO_THREAD_THRESHOLD` is parsed with `asyncio.to_thread` instead of inline, so a large
  body's CPU-bound split/decode work doesn't block the event loop (review 2 finding #9).
- `multipart/form-data` goes through `_StrictMultiPartParser`, a thin subclass of
  Starlette's own `MultiPartParser` (its `on_part_end` silently re-decodes an invalid UTF-8
  text part as latin-1 instead of rejecting it -- `_user_safe_decode` in
  `starlette/formparsers.py`).

Both paths reject a declared charset other than `utf-8`/`utf8` (case-insensitive, review 2
finding #6). Every field except `ref_audio` must be text (review 1 finding #1); `ref_audio`
must be a file part, not text, wherever and however many times it's given -- multipart is
checked one part at a time in wire order so a text-then-file `ref_audio` can't hide behind a
later, valid file part (review 2 finding #3; `form.get` alone only sees the *last* same-named
entry). A client-controlled field name is truncated before it's echoed into an error message
(review 2 finding #7). `ref_audio`'s bytes are read bounded to `MAX_AUDIO_BYTES + 1`, and
every `UploadFile`/`FormData` this module touches is closed before `read_fields` returns --
`Fields` only ever holds the plain strings and bytes actually extracted, never Starlette's
own form or query objects.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import parse_qsl, unquote_to_bytes

from fastapi import Request
from python_multipart.multipart import parse_options_header
from starlette.datastructures import FormData, QueryParams, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from breeze_infer import voice_file
from breeze_infer.errors import ApiError
from breeze_infer.limits import (
    MAX_AUDIO_BYTES,
    MAX_INSTRUCTION_CHARS,
    MAX_NEW_TOKENS_CEILING,
    MAX_REF_TEXT_CHARS,
    MAX_TEXT_CHARS,
)
from breeze_infer.settings import Settings
from breeze_infer.text_rules import has_control_characters
from breeze_infer.text_split import speakable

# contracts/http-api.md "Fields": the defaults for POST /v1/audio/speech.
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
DEFAULT_CFG_SCALE = 1.0
DEFAULT_SEED = 42

# Each sampling/reference field's [low, high] bound (data-model.md SpeechRequest), shared
# with ws_messages.parse (T075) so HTTP and WebSocket can't drift apart. Decimal bounds
# are literal text, not `float`, for the same exactness reason decimal_literal_in_range
# takes `low`/`high` as strings. `low_inclusive` and FR-006's zero-sentinel stay at each
# call site -- they differ per field.
CFG_SCALE_RANGE = ("0", "100")
TEMPERATURE_RANGE = ("0", "10")
TOP_P_RANGE = ("0", "1")
REPETITION_PENALTY_RANGE = ("0.0001", "10")
SEED_RANGE = (0, 4_294_967_295)
SPLIT_CHARS_RANGE = (0, 10_000)
TOP_K_RANGE = (1, 10_000)
MAX_NEW_TOKENS_RANGE = (1, MAX_NEW_TOKENS_CEILING)

# The limits passed to the multipart parser, and to _urlencoded_pairs_sync by hand
# (research.md R6). Letting Starlette's own parser accept a few file parts, not just one, is
# what lets `read_fields`'s own BC-08 pass (`_check_no_duplicate_names`, run over
# `form.multi_items()` -- `_split_multipart_fields` no longer tells duplicates apart itself;
# BC-08 has already ruled a repeated name out by the time it's called) see "a second (or
# third) ref_audio file" for itself, rather than Starlette's own parser refusing to hand back
# that many file parts at all: with max_files still 1, any file past the first would never
# reach this module's own code to be told apart from anything, and would surface as the
# generic parser error instead of naming ref_audio specifically. 4 is enough headroom for a
# client that sends a handful of duplicates by mistake (or to probe for this); a client
# sending *more* than that trips Starlette's own file-count limit before parsing even
# finishes -- `_parse_multipart_form` recovers `duplicate_field` from that case too, by
# running BC-08's own duplicate check over whatever names the parser had already seen (T048
# post-final review finding #3), and falls back to the generic error when there's no
# duplicate to find there, since the exception itself carries no field name to be more
# specific with. `FORM_MAX_FILES` still exists to bound how much of this parsing work a
# single request can ask Starlette to do at all.
FORM_MAX_FILES = 4
FORM_MAX_FIELDS = 32
# T037 review 1 finding #3, review 2 finding #4: a 4-byte UTF-8 code point (an emoji, or a
# CJK Extension-B character) percent-encodes to 12 ASCII bytes ("%XX" x 4), so a urlencoded
# `text` field at the full MAX_TEXT_CHARS (10,000) needs up to 120,000 bytes on the wire.
# Doubled from that exact figure to leave real headroom: the field-level length check
# (T048's `text_too_long`) must always be the one that actually rejects an over-length
# `text`, never this module's own generic parser-error message arriving first by a
# coincidence of exact boundaries.
FORM_MAX_PART_SIZE = 2 * MAX_TEXT_CHARS * 12

# review 2 finding #9: a body (or query string) larger than this is parsed off the event
# loop, via asyncio.to_thread, rather than inline.
_TO_THREAD_THRESHOLD = 64 * 1024

# review 2 finding #6: the only charset this contract accepts, spelled either way,
# case-insensitively.
_UTF8_CHARSET_NAMES = frozenset({"utf-8", "utf8"})

# review 2 finding #7: how much of a client-controlled field name survives into an error
# message before it's truncated.
_FIELD_LABEL_MAX_CHARS = 64


@dataclass(frozen=True)
class Fields:
    """The request's fields, already reduced to plain strings and bytes.

    Never Starlette's `FormData` or `QueryParams` straight from `request.query_params`
    (review 1 finding #5, review 2 finding #5): `FormData` holds each `UploadFile`'s own
    open `SpooledTemporaryFile` (closed before `read_fields` returns -- there'd be nothing
    left downstream to read from even if something tried), and `request.query_params` is
    Starlette's own lenient parse (silently `errors="replace"`), not this module's strict
    one. `form` and `query` here are both freshly built `QueryParams` over already-decoded,
    already-validated strings.

    `form` and `query` are kept apart, rather than merged into one mapping, purely so a
    caller reading a field (`_first`) can tell which source's `getlist` to consult -- by
    construction time, `read_fields`'s own `_check_no_duplicate_names` has already ruled out
    either one holding more than one value for the same name (research.md R6, BC-08).
    """

    form: QueryParams
    query: QueryParams
    ref_audio: bytes | None


async def read_fields(request: Request) -> Fields:
    """Read the body (bounded per research.md R6) and merge it with the query string.

    The query string is parsed by the same strict routine as a urlencoded body (review 2
    finding #5), from `request.scope["query_string"]` -- the raw wire bytes, not
    `request.query_params` (Starlette's own lenient parse).

    Two checks need nothing from the body at all -- a duplicate wholly within the query
    string, and `ref_audio` given as a query parameter -- so they run first, before the
    request's content type is even sniffed (T048 post-final review findings #4/#7). Without
    this, a rejected content type, an unsupported charset, or a large multipart file being
    spooled to disk could each surface their own error ahead of a `ref_audio` that was
    always going to be rejected regardless of what the body turned out to hold:
    `?ref_audio=x` with a `text/plain` body used to get the content-type error instead of
    `ref_audio must be a file part`; the same query with a declared-latin-1 urlencoded body
    used to get "must be UTF-8" instead; and a large multipart file used to be parsed and
    spooled before a query-only `ref_audio` was ever rejected.

    Routes on the request's declared media type: `multipart/form-data` and
    `application/x-www-form-urlencoded` are parsed as the module docstring describes;
    anything else is `400 invalid_field` *unless* the body is empty -- a query-only request
    with no body has no content type worth trusting, and still has to work (review 1
    finding #7, review 2 finding #8).

    Each of the multipart and urlencoded branches below then produces its own `(names,
    pairs)` from the body, and runs BC-08's duplicate check once more, combined with the
    query's own names (`_check_no_duplicate_names`) -- this is what catches a name repeated
    *across* the query and the body, not just within one source alone -- followed by that
    branch's own ref_audio-as-text check (`_reject_ref_audio_text` for urlencoded; multipart
    has no separate call at all, since its own per-part type check in
    `_split_multipart_fields` does the equivalent job in wire order, review 2 finding #3).
    The no-body branch runs neither a second time: the query-only checks above already
    covered everything a request with no body could still have wrong. Both checks, where
    they run, come before any other per-field check, `parse_speech`'s included: a request
    whose `text` happens to be too long but whose unrelated `seed` is duplicated across the
    query and the body must still get `duplicate_field`, not `text_too_long` -- *unless*
    `ref_audio` was already rejected from the query alone first: `?ref_audio=x&seed=1` with a
    body `seed=2` is `ref_audio must be a file part`, not `duplicate_field`, since the
    query-only `ref_audio` check above runs before the body is ever combined into that second
    duplicate pass (T048 post-final review finding #8).
    """
    query_pairs = await _parse_urlencoded_bytes(request.scope["query_string"])
    query_names = [name for name, _ in query_pairs]
    _check_no_duplicate_names(query_names)
    _reject_ref_audio_text(query_pairs)

    media_type, charset = _content_type(request)
    if media_type == "multipart/form-data":
        form = await _parse_multipart_form(request, query_names)
        try:
            raw_items = list(form.multi_items())
            _check_no_duplicate_names(query_names + [name for name, _ in raw_items])
            pairs, ref_audio_part = _split_multipart_fields(raw_items)
            ref_audio = await _read_ref_audio_bytes(ref_audio_part)
        finally:
            # Every UploadFile this form holds -- ref_audio's, and any file wrongly sent
            # under another field's name -- is closed before returning. `Fields` never
            # holds `form`.
            await form.close()
    elif media_type == "application/x-www-form-urlencoded":
        if not _is_utf8_charset(charset):
            raise ApiError(400, "invalid_field", "request body must be UTF-8 text")
        body = await request.body()  # already bounded by BodyLimitMiddleware
        pairs = await _parse_urlencoded_bytes(body, query_names)
        _check_no_duplicate_names(query_names + [name for name, _ in pairs])
        _reject_ref_audio_text(pairs)
        ref_audio = None
    else:
        if await _has_body(request):
            raise ApiError(
                400,
                "invalid_field",
                "content type must be multipart/form-data or "
                "application/x-www-form-urlencoded",
            )
        # The query-only checks above already covered everything there is to check here --
        # the body is empty, so there are no more names to combine and no body-side
        # ref_audio-as-text to reject.
        pairs, ref_audio = [], None

    return Fields(form=QueryParams(pairs), query=QueryParams(query_pairs), ref_audio=ref_audio)


def _content_type(request: Request) -> tuple[str, str]:
    """The request's Content-Type media type and charset (default `utf-8`), both
    lowercased; the media type carries no parameters."""
    header = request.headers.get("content-type", "")
    media_type, params = parse_options_header(header)
    charset = params.get(b"charset", b"utf-8")
    return media_type.decode("latin-1").lower(), charset.decode("latin-1").lower()


def _is_utf8_charset(charset: str) -> bool:
    """review 2 finding #6: accept `utf-8`/`utf8` case-insensitively, reject anything
    else declared (multipart's per-part charset, or the urlencoded Content-Type's)."""
    return charset.lower() in _UTF8_CHARSET_NAMES


def _label(name: str) -> str:
    """Truncate a client-controlled field name before it's echoed into an error message
    (review 2 finding #7) -- nothing bounds a field *name*'s length yet (T048 adds a length
    check to field *values*), so without this an arbitrarily long name could make the error
    response itself arbitrarily large."""
    if len(name) <= _FIELD_LABEL_MAX_CHARS:
        return name
    return name[:_FIELD_LABEL_MAX_CHARS] + "…"


def _display_label(raw: bytes) -> str:
    """A label for a field name that failed to decode as UTF-8 itself: latin-1 never
    fails, so this always produces *something* to show, truncated the same way."""
    return _label(raw.decode("latin-1"))


async def _has_body(request: Request) -> bool:
    """True if the request has a non-empty body, read only up to the first chunk (review 2
    finding #8) -- rejecting an unsupported content type shouldn't first buffer an
    arbitrarily large body just to learn that it's non-empty. `Request.stream()` only ever
    yields a non-empty chunk for real body data, with one trailing `b""` at the true end
    (`starlette/requests.py`), so the first item it yields already answers this.
    """
    async for chunk in request.stream():
        return bool(chunk)
    return False


def _reject_ref_audio_text(pairs: list[tuple[str, str]]) -> None:
    """`ref_audio` can only be a multipart file part, so a same-named field anywhere else
    (a urlencoded field, or a query parameter) is rejected the same way a string multipart
    field is (`_split_multipart_fields` below)."""
    if any(name == "ref_audio" for name, _ in pairs):
        raise ApiError(400, "invalid_field", "ref_audio must be a file part")


async def _parse_urlencoded_bytes(
    data: bytes, other_names: list[str] | None = None
) -> list[tuple[str, str]]:
    """`data` (a request body or `scope["query_string"]`), parsed and strictly UTF-8
    decoded. Runs inline for a small payload, or off the event loop for a large one
    (review 2 finding #9) -- `_urlencoded_pairs_sync` is plain CPU-bound work either way.

    `other_names` -- the query string's own names, when `data` is a request body rather than
    the query string itself -- is only ever used by `_urlencoded_pairs_sync`'s own
    `except ValueError` fallback (T048 post-final review finding #3); it plays no part when
    parsing succeeds.
    """
    if len(data) > _TO_THREAD_THRESHOLD:
        return await asyncio.to_thread(_urlencoded_pairs_sync, data, other_names)
    return _urlencoded_pairs_sync(data, other_names)


def _urlencoded_pairs_sync(
    body: bytes, other_names: list[str] | None = None
) -> list[tuple[str, str]]:
    """The synchronous work behind `_parse_urlencoded_bytes` (review 2 findings #1/#2/#9):
    split on `parse_qsl`, then UTF-8 decode every name and value by hand -- see the module
    docstring for why `parse_qsl`'s own `encoding`/`errors` parameters can't be trusted to
    do that for a `bytes` input.
    """
    if not body:
        return []
    try:
        raw_pairs = parse_qsl(body, keep_blank_values=True, max_num_fields=FORM_MAX_FIELDS)
    except ValueError:
        # parse_qsl's own "Max number of fields exceeded" guard: one O(len(body)) count of
        # the separator byte, raised before any per-field split or decode work starts, so a
        # huge body with many tiny fields is rejected in roughly one pass over the bytes
        # (test_a_huge_body_of_many_pairs_fails_fast_on_the_field_cap,
        # test_a_huge_body_of_distinct_names_fails_fast_on_the_field_cap) -- that guarantee
        # has to survive the fallback below too, so `_names_from_raw_urlencoded` bounds its
        # own split to at most `FORM_MAX_FIELDS + 1` pieces, never the whole body (T048
        # post-final review finding #2, HIGH). T048 post-final review finding #3: a name
        # repeated enough times to have caused this (33 `seed=`... fields, say) is
        # `duplicate_field`, a more useful answer than this generic error, which names no
        # field at all -- `_check_no_duplicate_names` raises that itself when it finds one;
        # otherwise this falls through to the same generic error as before.
        names = list(other_names) if other_names else []
        _check_no_duplicate_names(names + _names_from_raw_urlencoded(body))
        raise ApiError(400, "invalid_field", "could not parse the request body") from None

    pairs: list[tuple[str, str]] = []
    for raw_name, raw_value in raw_pairs:
        if len(raw_name) + len(raw_value) > FORM_MAX_PART_SIZE:
            # review 2 finding #7: the name counts against the bound too, not just the
            # value -- a giant name could otherwise dodge this check entirely.
            raise ApiError(400, "invalid_field", "could not parse the request body")
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            raise ApiError(
                400, "invalid_field", f"{_display_label(raw_name)} must be UTF-8 text"
            ) from None
        try:
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError:
            raise ApiError(
                400, "invalid_field", f"{_label(name)} must be UTF-8 text"
            ) from None
        pairs.append((name, value))
    return pairs


def _names_from_raw_urlencoded(body: bytes) -> list[str]:
    """The field names among the first `FORM_MAX_FIELDS + 1` pieces `body` splits into on
    `&` (`parse_qsl`'s own separator) -- capped there, not split in full, since `parse_qsl`'s
    own field-count guard only ever fires once there are at least that many pieces (T048
    post-final review finding #2, HIGH -- a DoS: an earlier version split the *whole* body
    and ran `Counter` over every piece, ~1.25s and ~1.1GB RSS with the GIL held for a 26 MiB
    body of two-byte names; the sibling test using single-letter names only passed because
    CPython caches those as small-string singletons,
    `test_a_huge_body_of_many_pairs_fails_fast_on_the_field_cap`).
    `body.split(b"&", FORM_MAX_FIELDS + 1)` performs at most that many splits, so its own
    cost is bounded the same way regardless of how much of `body` follows the piece that
    trips the limit; the trailing remainder past that point (index `FORM_MAX_FIELDS + 1`,
    present only when `body` actually has more pieces than that) is sliced off rather than
    treated as one more name.

    Each name is then decoded exactly as `parse_qsl`'s own bytes branch would -- `+` means
    space, then percent-unescaped (`unquote_to_bytes`), then strict UTF-8 -- not left as raw
    wire bytes: comparing an undecoded name against `other_names` (the query's own,
    already-decoded names, from `_urlencoded_pairs_sync`'s caller) used to risk a false
    duplicate (`?a%2Bb=1`, decoded name `a+b`, next to a body field literally spelled `a+b`,
    decoded name `a b` -- different names that only looked equal undecoded) or a missed real
    one (`seed` next to `se%65d`, the same name once both are decoded) (T048 post-final
    review findings #4/#5). A name that fails to decode as UTF-8 raises the same `400
    invalid_field` the successful parse path (`_urlencoded_pairs_sync`) would have raised for
    it, rather than being silently papered over here -- decoding only the first
    `FORM_MAX_FIELDS + 1` names keeps this as cheap as the rest of this function.
    """
    pieces = body.split(b"&", FORM_MAX_FIELDS + 1)[: FORM_MAX_FIELDS + 1]
    names: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        raw_name = piece.split(b"=", 1)[0]
        try:
            names.append(unquote_to_bytes(raw_name.replace(b"+", b" ")).decode("utf-8"))
        except UnicodeDecodeError:
            raise ApiError(
                400, "invalid_field", f"{_display_label(raw_name)} must be UTF-8 text"
            ) from None
    return names


class _StrictMultiPartParser(MultiPartParser):
    """Starlette's own `MultiPartParser.on_part_end` (`starlette/formparsers.py`) decodes a
    text part with `_user_safe_decode`, which silently falls back to a latin-1 decode when
    the part isn't valid under its charset -- so a genuinely corrupt UTF-8 part is never
    rejected, just mojibake'd. This overrides only that one callback: the declared charset
    must be UTF-8 (`_is_utf8_charset`, review 2 finding #6), and the bytes must actually
    decode as `utf-8` with `errors="strict"`, or the field is rejected the same way every
    other field-level error in this module is. Everything else -- boundary parsing, file
    spooling, the file/field count and size limits -- is untouched Starlette machinery,
    reached via `super()`.
    """

    def on_part_end(self) -> None:
        if self._current_part.file is not None:
            super().on_part_end()
            return
        field_name = self._current_part.field_name
        if not _is_utf8_charset(self._charset):
            raise ApiError(400, "invalid_field", f"{_label(field_name)} must be UTF-8 text")
        try:
            value = self._current_part.data.decode("utf-8")
        except UnicodeDecodeError:
            raise ApiError(
                400, "invalid_field", f"{_label(field_name)} must be UTF-8 text"
            ) from None
        self.items.append((field_name, value))


async def _parse_multipart_form(request: Request, query_names: list[str]) -> FormData:
    """`multipart/form-data`, via `_StrictMultiPartParser` -- not `request.form()`, which
    constructs the base (silently-lossy) parser and gives no way to swap it in.

    Mirrors what `Request._get_form` does for this content type (`starlette/requests.py`):
    the same limits, and the same fallback for a boundary/limit error (`MultiPartException`,
    Starlette's own class in `formparsers.py` -- unrelated to
    `python_multipart.exceptions.FormParserError`, which `errors.py` already maps to `400`
    for the lower-level parser's own malformed-syntax errors, which still propagate
    unchanged since this module never catches them).

    Only parses -- doesn't validate which parts are text and which are `ref_audio`, and
    doesn't check for duplicates: `read_fields` needs the raw `(name, value)` pairs this
    produces (via `form.multi_items()`) to run the duplicate-field pass first, over every
    name from both the query string and this form, before any of that per-field validation
    (`_split_multipart_fields`) runs. The caller owns `form`'s lifetime (`await
    form.close()`), same as before.

    A `MultiPartException` raised for hitting a part, field or file limit never reaches that
    BC-08 pass at all -- parsing stopped before `form.multi_items()` ever existed to run it
    over. `query_names` is threaded through just for this case: on that exception,
    `_names_seen_by_multipart_parser` recovers whatever names the parser *had* already seen,
    and `_check_no_duplicate_names` (the same BC-08 check, just fed an incomplete list) still
    catches a name repeated among them -- five or more `ref_audio` file parts, say -- as
    `duplicate_field`, rather than leaving the client with the generic "could not parse"
    error, which wouldn't name the field at all (T048 post-final review finding #3).
    """
    parser = _StrictMultiPartParser(
        request.headers,
        request.stream(),
        max_files=FORM_MAX_FILES,
        max_fields=FORM_MAX_FIELDS,
        max_part_size=FORM_MAX_PART_SIZE,
    )
    try:
        return await parser.parse()
    except MultiPartException as exc:
        _check_no_duplicate_names(query_names + _names_seen_by_multipart_parser(parser))
        raise ApiError(400, "invalid_field", "could not parse the request body") from exc


def _names_seen_by_multipart_parser(parser: _StrictMultiPartParser) -> list[str]:
    """Every field name `parser` had already produced (`parser.items`, every part that ran
    all the way through `on_part_end`), plus the one it was in the middle of when a
    `MultiPartException` -- a part, field or file limit -- cut it off
    (`parser._current_part.field_name`, set by `on_headers_finished` before it checks either
    limit, so it's there for a field-count trip exactly as it is for a file-count one).

    Both are read through `getattr`, defaulting to "nothing seen" rather than raising:
    neither is a stable, documented Starlette attribute (`items` is closer to one than
    `_current_part` is, but this module doesn't rely on either staying put), so a future
    Starlette release renaming or dropping either one should fall back to
    `_parse_multipart_form`'s own generic error, not a 500 (T048 post-final review
    finding #3).

    `current_name` is appended only when it's truthy, not merely "not `None`":
    `MultipartPart.field_name` (`starlette/formparsers.py`) defaults to `""`, not `None`, so
    a part that never got as far as having a real name read from it (a request whose
    multipart parsing fails before any part does, e.g. a missing boundary) still has a
    `_current_part` with that empty default -- appending it unconditionally used to make an
    unrelated empty-named query field (`?=1`) look like the same field given twice, `400
    duplicate_field` naming no field at all (T048 post-final review finding #3, review 31b).
    """
    names = [name for name, _ in getattr(parser, "items", [])]
    current_part = getattr(parser, "_current_part", None)
    current_name = getattr(current_part, "field_name", None)
    if current_name:
        names.append(current_name)
    return names


def _split_multipart_fields(
    raw_items: list[tuple[str, str | UploadFile]],
) -> tuple[list[tuple[str, str]], UploadFile | None]:
    """Split `raw_items` into ordinary text fields and (at most one) `ref_audio` file part.

    By the time `read_fields` calls this, BC-08's duplicate-field pass
    (`_check_no_duplicate_names`) has already rejected any name given more than once --
    `ref_audio` included -- as `400 duplicate_field` (T048 post-final review finding #6: a
    *second* `ref_audio` part, file or text, never reaches this loop at all). So this is a
    single pass over parts already known to have at most one of any given name: the one
    `ref_audio` part, if there is one, must be a file; every other part must be text (review
    1 finding #1) -- a file part sent under any other field name is rejected outright.
    """
    pairs: list[tuple[str, str]] = []
    ref_audio_part: UploadFile | None = None
    for name, value in raw_items:
        if name == "ref_audio":
            if not isinstance(value, UploadFile):
                raise ApiError(400, "invalid_field", "ref_audio must be a file part")
            ref_audio_part = value
            continue
        if isinstance(value, UploadFile):
            raise ApiError(400, "invalid_field", f"{_label(name)} must be a text field")
        pairs.append((name, value))
    return pairs, ref_audio_part


async def _read_ref_audio_bytes(part: UploadFile | None) -> bytes | None:
    """A real `ref_audio` file part is read bounded to one byte past `MAX_AUDIO_BYTES` --
    enough for `reference_audio.decode()` to tell "too big" from "right at the limit" later
    without this module ever buffering more than that itself."""
    if part is None:
        return None
    return await part.read(MAX_AUDIO_BYTES + 1)


def _check_no_duplicate_names(names: list[str]) -> None:
    """BC-08: every key given more than once -- across the query string and the body
    combined, known to this contract or not (`foo=1&foo=2` is `400 duplicate_field` even
    though `foo` isn't a field `parse_speech` ever reads) -- is `400 duplicate_field` before
    any per-field check runs. `read_fields` calls this *twice*: once over the query string
    alone, before the body is even sniffed (query-only duplicates, and `ref_audio` given as a
    query parameter, are knowable without it); once more per body branch, over the query's
    names combined with that branch's -- this second call is what catches a name repeated
    *across* the query and the body, not just within one source alone. Both calls come before
    `_reject_ref_audio_text`, `_split_multipart_fields`'s own file-vs-text check, and every
    check in `parse_speech`. `_parse_multipart_form` and `_urlencoded_pairs_sync` also call
    this, over whatever names a parsing limit left them, when a `MultiPartException` or
    `parse_qsl`'s own field-count guard cuts a parse short (T048 post-final review finding
    #3) -- same function, same rule, just fed a different, incomplete set of names.

    `names` is the plain list of every name given to a single call, in wire order, with
    repeats -- not deduplicated first -- so a name repeated only within one source (twice in
    the query, say) is counted the same way as one split across both. `Counter` finds
    whichever of those is true for a given name; the loop over `names` (not `counts`) keeps
    the error naming whichever duplicated name appears *first* on the wire, deterministically.
    """
    counts = Counter(names)
    for name in names:
        if counts[name] > 1:
            raise ApiError(400, "duplicate_field", f"{_label(name)} was given more than once")


def _first(fields: Fields, name: str) -> str | None:
    """The field's value, or `None` when it's absent or empty (BC-02).

    Duplicate detection (BC-08) already ran, twice, before `Fields` was ever built -- once
    over the query string alone, once more over the query's names combined with the body's
    (`_check_no_duplicate_names`, called by `read_fields`) -- so, by the time this runs,
    `name` is already known to have at most one value between the form and the query string,
    and this is purely "the one value, empty means absent".
    """
    values = [*fields.form.getlist(name), *fields.query.getlist(name)]
    if not values or values[0] == "":
        return None
    return values[0]


# contracts/http-api.md's number grammar, compiled once. `re.ASCII` is required, not
# decorative: Python's bare `\d` matches every Unicode decimal-digit character (e.g. an
# Arabic-Indic digit), which `int()`/`float()` then happily parse too -- `re.ASCII` restricts
# `\d` to `[0-9]`, so a non-ASCII numeral fails the grammar the same as any other non-numeral
# text. `fullmatch` is used throughout rather than the contract's literal `^...$` spelling:
# in Python, a bare `$` also matches just before one trailing newline, which would let
# "12\n" slip through where the contract means it not to.
#
# DECISION (final review): the exponent is *not* capped -- the contract's own grammar
# (`^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`) allows any number of exponent digits, and nothing
# here should be stricter than the contract itself. A literal with an exponent
# `decimal.Decimal` can't represent (`MAX_EMAX`/`MIN_EMIN`, +/-999999999999999999 -- not the
# active context's Emax/Emin, roughly +/-999999, which only bounds arithmetic, not
# construction; T048 post-final review finding #6) -- e.g. `Decimal("1e1000000000000000000")`
# -- is instead handled by `_check_decimal_range`
# below, which catches `decimal.InvalidOperation` directly and rebuilds a `Decimal` with the
# literal's own sign and digits, but a re-anchored exponent (`_decimal_from_unrepresentable_
# literal`, T048 post-final review finding #1) -- not the already-parsed `float`, whose
# underflow-to-zero loses the sign a range check needs. For the optional sampling fields,
# `is_zero_literal` means a zero-mantissa literal like `0e99999` never reaches `Decimal`
# construction at all -- but that's *not* true of every numeric field: `cfg_scale` has no
# such pre-check (0 is an ordinary in-range value for it, not a "use the default" sentinel),
# so `cfg_scale=0e1000000000000000000` genuinely does reach, and overflow, `Decimal`'s
# context, and relies on `_check_decimal_range`'s fallback to still be judged correctly.
_INT_LITERAL = re.compile(r"[+-]?\d+", re.ASCII)
_DECIMAL_LITERAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)

# Python 3.11+ refuses to convert a digit string longer than
# `sys.get_int_max_str_digits()` (4,300 by default) to `int` at all -- it raises
# `ValueError`, unhandled, which reached the client as a bare `500`. No field this contract
# validates ever needs more than 10 digits (the widest range, seed, tops out at
# 4294967295); 20 is a generous margin that's still nowhere near the CPython limit, so a
# literal this long is rejected as out-of-range before `int()` is ever called on it, rather
# than relying on a limit that exists for a different reason and could itself change.
MAX_INT_LITERAL_DIGITS = 20


def significant_int_digits(value: str) -> tuple[str, str]:
    """`value`'s sign and significant digits, leading zeros stripped, so a padded
    literal (`"0" * 25 + "1"`) is judged by its actual magnitude. `value` must already be
    known to match `_INT_LITERAL` (or JSON's own integer grammar, a subset of it) --
    this only strips characters, it never validates them. Shared by `_parse_int` below
    and by `ws_messages.py`'s integer fields.
    """
    sign = "-" if value.startswith("-") else ""
    significant_digits = value.lstrip("+-").lstrip("0") or "0"
    return sign, significant_digits


def _parse_int(value: str, field: str, rule: str) -> int:
    """The strict integer grammar (BC-01): reject anything `int()` itself is more lenient
    about than the contract is -- surrounding whitespace, a non-ASCII digit, underscore
    digit separators -- before ever calling `int()`, so only a string already known to
    match `^[+-]?\\d+$` reaches it.

    `rule` is the field's own range wording, used only for the digit-count guard above: a
    literal that grammar-matched (it *is* all digits) but is absurdly long is out of range,
    not malformed, so it gets that field's `"<field> must be <rule>"` message, not the
    generic "must be an integer" one.

    Regression (final review): `int()` is called on `sign + significant_digits`, the
    stripped result, never on the original `value`. Calling it on `value` instead -- even
    after the digit count above had already confirmed the *significant* digit count was
    within bounds -- would still hand `int()` the original, unstripped literal: for
    something like `"0" * 5000 + "1"`, over `sys.get_int_max_str_digits()` (4,300 by
    default) raw digits even though its value is just `1`, that call alone raises
    `ValueError`, unhandled, the same `500` this whole digit cap exists to prevent.
    """
    if _INT_LITERAL.fullmatch(value) is None:
        raise ApiError(400, "invalid_field", f"{field} must be an integer")
    sign, significant_digits = significant_int_digits(value)
    if len(significant_digits) > MAX_INT_LITERAL_DIGITS:
        raise ApiError(400, "invalid_field", f"{field} must be {rule}")
    return int(sign + significant_digits)


def _parse_decimal(value: str, field: str) -> float:
    """The strict decimal grammar (BC-01): same idea as `_parse_int`, for
    `^[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?$` -- this also rejects `inf` and `nan`,
    which Python's own `float()` accepts but the grammar's digit-only pattern never matches.

    The result is not yet range- or finiteness-checked -- `_check_decimal_range` and the
    `cfg_scale` call site in `parse_speech` do that next. A value that grammar-matches but
    overflows to `inf` (e.g. `1e400`) is caught there, not here: "finite" is a range
    concern, not a grammar one.
    """
    if _DECIMAL_LITERAL.fullmatch(value) is None:
        raise ApiError(400, "invalid_field", f"{field} must be a number")
    return float(value)


def is_zero_literal(value: str) -> bool:
    """Whether a decimal literal already known to match `_DECIMAL_LITERAL` is zero,
    decided from its digits rather than its parsed `float` value.

    Review finding: `float("1e-400")` underflows to exactly `0.0`, even though its only
    digit is `1` -- treating that as the "0 means use the model default" sentinel would
    silently accept a value the field's own range actually rejects (a temperature of
    `1e-400` is not `0`; it is an out-of-range positive number that happens to underflow).
    Stripping the optional sign, the decimal point and the exponent leaves just the
    mantissa's digits; the literal is zero only when every one of them is `0`.
    """
    mantissa = value.split("e", 1)[0].split("E", 1)[0]
    digits = mantissa.lstrip("+-").replace(".", "")
    return set(digits) == {"0"}


# T048 post-final review finding #1: how far outside decimal's own representable range
# (`decimal.MAX_EMAX`/`MIN_EMIN`, +/-999999999999999999 -- *not* the active context's
# Emax/Emin, roughly +/-999999, which only bounds *arithmetic*, not construction; T048
# post-final review finding #6 (NIT): `Decimal("1e999999999999999999")` constructs fine,
# `InvalidOperation` only starts one exponent digit further out) `_decimal_from_
# unrepresentable_literal` re-anchors an unrepresentable literal's exponent. Any magnitude
# comfortably past every field this contract validates (all bounded between 0.0001 and
# 4294967295) works equally well; this is comfortably past decimal's own representable range
# too, so the reconstructed Decimal is itself never rejected by anything downstream.
_CLAMPED_EXPONENT = 10**6

# T048 post-final review finding #1 (HIGH, a 500 regression): past this many significant
# exponent digits, `_parse_exponent` clamps straight to `_CLAMPED_EXPONENT` rather than
# calling `int()` at all -- Python 3.11+ refuses to convert a digit string longer than
# `sys.get_int_max_str_digits()` (4,300 by default) to `int`, raising `ValueError`,
# unhandled, for a literal like `cfg_scale=1e` + 5,000 nines. A comfortable margin under
# that (not right up against it, since digit count alone doesn't bound the *value* -- an
# 18-digit exponent can already exceed `MAX_EMAX`) is all this needs: any exponent this long
# is already far past `_CLAMPED_EXPONENT` in magnitude, so the clamp is exact either way.
_MAX_SAFE_EXPONENT_DIGITS = 20


def _parse_exponent(exponent_digits: str) -> int:
    """The value of a decimal literal's exponent digits (the part after `e`/`E`, sign
    included, from `_DECIMAL_LITERAL`'s own grammar) -- or, once there are more than
    `_MAX_SAFE_EXPONENT_DIGITS` of them, `_CLAMPED_EXPONENT` with the same sign, computed
    without ever calling `int()` on the full string (T048 post-final review finding #1).

    Leading zeros are stripped *before* counting digits, not after: an exponent padded with
    thousands of zeros ahead of a single significant digit (`e` + 5,000 zeros + `1`) is
    genuinely just `1`, cheap and safe to convert, not a huge number that happens to look
    short once reduced -- counting raw digits would clamp that too, silently turning an
    ordinary small value into a wildly wrong one.
    """
    if not exponent_digits:
        return 0
    sign = -1 if exponent_digits.startswith("-") else 1
    magnitude_digits = exponent_digits.lstrip("+-").lstrip("0") or "0"
    if len(magnitude_digits) > _MAX_SAFE_EXPONENT_DIGITS:
        return sign * _CLAMPED_EXPONENT
    return sign * int(magnitude_digits)


def _decimal_from_unrepresentable_literal(literal: str) -> Decimal:
    """`literal` matched `_DECIMAL_LITERAL` but its exponent overflows `decimal.Decimal`'s
    representable range -- `Decimal(literal)` itself already raised `InvalidOperation`,
    which is what sends `_check_decimal_range` here instead of building `decimal_value`
    directly.

    Rebuilt from the literal's own sign and digits, with only the exponent re-anchored to
    `_CLAMPED_EXPONENT`: `Decimal`'s tuple constructor -- unlike its string constructor --
    isn't bounds-checked against the context at all, so it never raises here, however far
    outside decimal's representable range the clamped exponent still is. A zero mantissa is
    exactly zero, whatever its exponent (`is_zero_literal`'s rule, kept here too since
    `cfg_scale` never calls it separately, unlike the optional sampling fields); a nonzero
    mantissa keeps its sign and a merely very large (not unrepresentable) exponent, so
    `_check_decimal_range`'s ordinary bound comparison judges it exactly as it would any
    other out-of-range value -- unlike falling back to `Decimal(float(literal))`, whose
    underflow-to-zero can't carry a negative sign a range check needs (T048 post-final
    review finding #1).
    """
    if is_zero_literal(literal):
        return Decimal(0)
    mantissa, _, exponent_digits = literal.partition("e")
    if not exponent_digits:
        mantissa, _, exponent_digits = literal.partition("E")
    sign_bit = 1 if mantissa.startswith("-") else 0
    int_part, _, frac_part = mantissa.lstrip("+-").partition(".")
    # Leading zeros don't affect the value once the exponent accounts for the decimal
    # point's position, so stripping them here is safe -- `is_zero_literal` above already
    # ruled out every digit being zero, so at least one significant digit survives.
    digits = (int_part + frac_part).lstrip("0")
    exponent = -len(frac_part) + _parse_exponent(exponent_digits)
    clamped_exponent = max(-_CLAMPED_EXPONENT, min(_CLAMPED_EXPONENT, exponent))
    return Decimal((sign_bit, tuple(int(d) for d in digits), clamped_exponent))


def _check_int_range(value: int, field: str, low: int, high: int, rule: str) -> None:
    if not (low <= value <= high):
        raise ApiError(400, "invalid_field", f"{field} must be {rule}")


def decimal_literal_in_range(
    literal: str, value: float, low: str, high: str, *, low_inclusive: bool
) -> bool:
    """The bound comparison checks `literal`, via `decimal.Decimal`, not just the parsed
    `float`. `float`'s limited (~15-17 significant digit) precision can round a genuinely
    out-of-range literal into range -- `float("10.0000000000000001") == 10.0` exactly, so a
    naive `value <= 10` would wrongly accept it. `decimal.Decimal` parses the literal
    exactly, so that comparison is exact.

    `low`/`high` are strings, not `float`/`int`, for the same reason: `Decimal(0.0001)`
    (from the *float* `0.0001`, which itself can't be represented exactly in binary) would
    reintroduce the very imprecision this function exists to avoid, where `Decimal("0.0001")`
    (from the literal digits) doesn't.

    The `Decimal` check alone isn't sufficient, though: `value` (`float(literal)`) is what
    every caller actually keeps and returns downstream, and `float` underflow means the two
    can disagree about which side of the boundary a tiny literal is on. `temperature=1e-400`
    is a genuinely positive `Decimal` -- `Decimal` comparison alone would call it in-range
    for `(0, 10]` -- but `float("1e-400")` underflows to exactly `0.0`, which is *not* in
    `(0, 10]`, and `0.0` is what this field would actually carry from here on (the runtime's
    own range check, aligned to this same table, would then raise on that stored `0.0` after
    the busy check -- a `500` this validation exists to prevent). So both the exact `Decimal`
    check and the ordinary `float` check must pass; either one failing is out of range. (A
    separate `math.isfinite(value)` guard isn't needed alongside them: an overflowed `float`
    -- `value == inf` -- already fails the plain `value <= float_high` comparison in
    `float_in_range` on its own, no special-casing required.)

    `Decimal(literal)` construction is guarded by `try`/`except InvalidOperation` as the
    *primary* handling for a literal the grammar accepts but `Decimal` can't represent
    (`MAX_EMAX`/`MIN_EMIN`, +/-999999999999999999 -- not the active context's Emax/Emin,
    roughly +/-999999, which bounds arithmetic, not construction; T048 post-final review
    finding #6) -- e.g. `cfg_scale=0e1000000000000000000` -- not a safety net alongside some
    other cap: the grammar's exponent cap was removed (see the DECISION above
    `_INT_LITERAL`), so this is now how such a literal is actually handled, for every numeric
    field this function guards. `cfg_scale` in particular never runs `is_zero_literal` first
    the way the optional sampling fields do (0 is an ordinary in-range value for it, not a
    "use the default" sentinel), so a zero-mantissa literal with an unrepresentable exponent
    reaches here, not just a nonzero one (T048 post-final review finding #1).

    On that exception, `decimal_value` is rebuilt by `_decimal_from_unrepresentable_literal`
    from the literal's own sign and digits, not from `Decimal(value)` (the already-parsed
    `float`). That used to be the fallback, and it loses information a range check needs:
    `float`'s underflow only ever produces `+0.0` or `-0.0`, and `Decimal(-0.0)` compares
    `>=` zero the same as `Decimal(0.0)` does (decimal's own equality treats `-0` and `0` as
    equal) -- so a genuinely negative, merely tiny `cfg_scale` (`cfg_scale=
    -1e-999999999999999999999`) was wrongly accepted as in-range at `0.0`, while the
    representable `cfg_scale=-1e-400` correctly wasn't (T048 post-final review finding #1).
    Reconstructing from the literal's own digits keeps the sign and the nonzero-ness a zero
    *float* can't carry, so the ordinary bound comparison below decides it exactly as it
    would any other value: a mantissa of all zeros is exactly zero, whatever its exponent
    (`is_zero_literal`'s own rule, since `cfg_scale` never calls it separately) -- in range
    for `cfg_scale`'s `[0, 100]`, out of range for a sampling field that reaches this at all,
    like `temperature`'s `(0, 10]`; a nonzero mantissa keeps its sign, so a tiny negative
    value is out of range for `cfg_scale` too, not mistaken for `-0.0`; a huge positive
    exponent stays huge and positive, never inside any of this contract's (all finite) upper
    bounds.
    """
    try:
        decimal_value = Decimal(literal)
    except InvalidOperation:
        decimal_value = _decimal_from_unrepresentable_literal(literal)
    low_bound = Decimal(low)
    high_bound = Decimal(high)
    decimal_in_range = (
        decimal_value >= low_bound if low_inclusive else decimal_value > low_bound
    ) and decimal_value <= high_bound

    float_low, float_high = float(low), float(high)
    float_in_range = (
        value >= float_low if low_inclusive else value > float_low
    ) and value <= float_high

    return decimal_in_range and float_in_range


def _check_decimal_range(
    literal: str,
    value: float,
    field: str,
    low: str,
    high: str,
    *,
    low_inclusive: bool,
    rule: str,
) -> None:
    """The HTTP wrapper over `decimal_literal_in_range` above (see its docstring for the
    exactness reasoning this whole check exists for): raises this field's own `400
    invalid_field` wording when the literal is out of range. `ws_messages.py` calls
    `decimal_literal_in_range` directly instead -- a WebSocket `error` event has no
    per-field prose to fill in here, only a `code`.
    """
    if not decimal_literal_in_range(literal, value, low, high, low_inclusive=low_inclusive):
        raise ApiError(400, "invalid_field", f"{field} must be {rule}")


def _optional_int(fields: Fields, name: str, low: int, high: int, rule: str) -> int | None:
    """FR-006: `0` (or an absent field) means the model default (`None`); otherwise the
    value must fall in `[low, high]`. Python's `int` has arbitrary precision -- no overflow
    or underflow -- so unlike `_optional_decimal` below, the sentinel is decided from the
    parsed value directly."""
    raw = _first(fields, name)
    if raw is None:
        return None
    value = _parse_int(raw, name, rule)
    if value == 0:
        return None
    _check_int_range(value, name, low, high, rule)
    return value


def _optional_decimal(
    fields: Fields, name: str, low: str, high: str, *, low_inclusive: bool, rule: str
) -> float | None:
    """FR-006: `0` (or an absent field) means the model default (`None`); otherwise the
    value must fall in the given range. The sentinel is decided from the literal text
    (`is_zero_literal`), not the parsed `float`, per the review finding `is_zero_literal`
    documents -- so a nonzero literal that underflows still reaches, and fails,
    `_check_decimal_range` instead of being mistaken for the default."""
    raw = _first(fields, name)
    if raw is None:
        return None
    value = _parse_decimal(raw, name)
    if is_zero_literal(raw):
        return None
    _check_decimal_range(raw, value, name, low, high, low_inclusive=low_inclusive, rule=rule)
    return value


def _check_no_control_characters(value: str, field: str) -> None:
    """BC-46, via the shared rule (`text_rules.has_control_characters`, which the voice
    file reader applies to a saved `ref_text` too)."""
    if has_control_characters(value):
        # contract wording, "<field> must be <rule>".
        raise ApiError(400, "invalid_field", f"{field} must be free of control characters")


def _validated_text_field(fields: Fields, name: str, max_chars: int) -> str | None:
    """`ref_text`'s value (`None` when absent, BC-02), control-character- and length-checked
    (BC-46/BC-05), in that order: control characters first, since `str.strip()` (and
    `str.isspace()`) treats several `Cc` control characters (`\\x1c`-`\\x1f`, `\\x85` NEL) as
    whitespace, so checking blank-ness first would let a value that's *only* one of those
    slip through as "blank" instead of being caught as the control-character violation it
    actually is; then blank (a whitespace-only value counts as absent here, the same as the
    general BC-02 "empty means absent"); only once it's known to be a real, non-blank value
    is its length checked -- a whitespace-only value of any length is still absent, not
    "too long". `instruction`, in `parse_speech`, applies this same order (control, then
    blank -- with its own meaning, the default -- then length); `text` is the one exception,
    checking length before control characters, since only it has a dedicated `text_too_long`
    code cheap enough to short-circuit an over-length scan for control characters.
    """
    raw = _first(fields, name)
    if raw is None:
        return None
    _check_no_control_characters(raw, name)
    if not raw.strip():
        return None
    if len(raw) > max_chars:
        raise ApiError(400, "invalid_field", f"{name} must be at most {max_chars:,} characters")
    return raw


@dataclass(frozen=True)
class NoReference:
    """No `voice_id` and no `ref_audio`: voice design."""


@dataclass(frozen=True)
class VoiceRef:
    """`voice_id`, with no `ref_audio`. `ref_text_override` replaces the stored
    transcript for this request only, when given."""

    voice_id: str
    ref_text_override: str | None


@dataclass(frozen=True)
class InlineRef:
    """`ref_audio` and `ref_text`, with no `voice_id`. `audio_bytes` is the raw upload;
    the route decodes it (FR-007 runs reference-consistency checks before the decode)."""

    audio_bytes: bytes
    ref_text: str


# data-model.md "ReferenceSpec": a tagged union over which reference, if any, the request
# names. `|` at module scope (not inside a `from __future__ import annotations`-deferred
# annotation) builds a real `types.UnionType`, usable both as a runtime value and as a type.
ReferenceSpec = NoReference | VoiceRef | InlineRef

# `voice_id` is a *lookup* key, so it must accept either shape a real voice id can have: a
# saved voice's name (contracts/http-api.md POST /v1/voices `name` field, checked by the
# one shared rule, `voice_file.is_valid_name`), or an unnamed voice's auto-generated `v_` +
# 16-lowercase-hex id (data-model.md). Whether the id actually exists is Phase 7's concern
# (the stub 404 lookup); this only checks its shape.
_VOICE_UNNAMED_ID_PATTERN = re.compile(r"v_[0-9a-f]{16}", re.ASCII)


def _is_valid_voice_id(value: str) -> bool:
    """Branches on the `v_` prefix rather than just checking the saved-name shape alone:
    that shape's character class would also accept a `v_`-prefixed string that isn't a
    real 16-hex id (e.g. `v_not-a-real-id`) as if it were a plausible saved name -- which it
    structurally can't be, since BC-26 forbids a saved name from ever starting with `v_`.

    The prefix check itself is case-insensitive (`value[:2].lower()`), not
    `value.startswith("v_")`: BC-26 makes saved names unique ignoring case, so the `v_`
    reservation has to be case-insensitive too, or `V_zz...` could sneak through as if it
    were an ordinary name just because its case doesn't literally match `v_`. Once routed
    into this branch, the match against `_VOICE_UNNAMED_ID_PATTERN` stays strictly
    case-sensitive: a real unnamed-voice id is always exactly lowercase, so `V_` followed by
    16 lowercase hex characters is still rejected here, same as before.

    Called through the module (`voice_file.is_valid_name`), not a bare imported name, so
    there is exactly one saved-name rule and a test can see this uses it.
    """
    if value[:2].lower() == "v_":
        return _VOICE_UNNAMED_ID_PATTERN.fullmatch(value) is not None
    return voice_file.is_valid_name(value)


def _build_reference(fields: Fields) -> ReferenceSpec:
    """data-model.md "ReferenceSpec": build the variant these fields describe.

    Checked in the contract's order (`reference_conflict`, then `ref_text_required`, then
    `reference_required`) even though T048 is what makes that order actually matter (it
    arrives once the reference fields have their own length/control-character checks that
    could otherwise race with these); doing it cheaply now costs nothing.

    `fields.ref_audio` is used as-is, not collapsed to `None` when empty: BC-02's "empty
    value means absent" is a rule about *fields* going missing from a form, not about a
    genuinely empty upload. An attached-but-empty `ref_audio` part is "present" for this
    variant-selection step, and reaches `reference_audio.decode()` later, which is what
    turns an empty blob into `400 invalid_audio` (data-model.md "DecodedAudio").
    """
    voice_id = _first(fields, "voice_id")
    if voice_id is not None and not _is_valid_voice_id(voice_id):
        # A field-syntax check (FR-007 order), so it runs before the reference-consistency
        # checks below, same as ref_text's own length/control-character check does.
        raise ApiError(400, "invalid_field", "voice_id must be a voice name or v_ id")
    ref_audio = fields.ref_audio
    # ref_text's own syntax (length, control characters) is checked here, before the
    # reference-consistency checks below, per FR-007's order: field syntax and ranges
    # (400) come before reference consistency (400).
    ref_text = _validated_text_field(fields, "ref_text", MAX_REF_TEXT_CHARS)

    if voice_id is not None and ref_audio is not None:
        raise ApiError(
            400, "reference_conflict", "voice_id and ref_audio cannot be used together"
        )
    if ref_audio is not None and ref_text is None:
        raise ApiError(400, "ref_text_required", "ref_text is required with ref_audio")
    if ref_text is not None and voice_id is None and ref_audio is None:
        raise ApiError(400, "reference_required", "ref_text needs ref_audio or voice_id")

    if ref_audio is not None:
        # ref_text is required whenever ref_audio is given (checked above), so it's never
        # None here.
        assert ref_text is not None
        return InlineRef(audio_bytes=ref_audio, ref_text=ref_text)
    if voice_id is not None:
        return VoiceRef(voice_id=voice_id, ref_text_override=ref_text)
    return NoReference()


@dataclass(frozen=True)
class SpeechRequest:
    """data-model.md "SpeechRequest": everything `POST /v1/audio/speech` needs, already
    validated. Produced by `parse_speech`; nothing downstream re-checks these fields."""

    text: str
    instruction: str
    reference: ReferenceSpec
    cfg_scale: float
    seed: int
    temperature: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    max_new_tokens: int | None
    split_chars: int


def parse_speech(fields: Fields, settings: Settings) -> SpeechRequest:
    """data-model.md "SpeechRequest": defaults, ranges, length limits, the control-
    character rule and FR-006's `0` -> `None` all applied, in the field's table order.

    Duplicate-field detection (BC-08) already ran, in `read_fields`, before this was ever
    called -- every field this function reads is already known to have at most one value.
    """
    text = _first(fields, "text")
    # DECISION (final review): text is the one field checked length-before-control -- a
    # cheap len() bound, before ever scanning the whole string for control characters -- but
    # the blank check (BC-10) still runs last, after both: str.strip() treats several Cc
    # control characters (\x1c-\x1f, \x85 NEL) as whitespace, so checking blank-ness before
    # control characters would let a `text` that is *only* one of those silently become
    # "blank" (text_required) instead of the BC-46 violation it actually is.
    if text is not None:
        if len(text) > MAX_TEXT_CHARS:  # BC-05
            raise ApiError(400, "text_too_long", "text is too long")
        _check_no_control_characters(text, "text")  # BC-46
    if text is None or not text.strip():
        raise ApiError(400, "text_required", "text is required")  # BC-10
    if not speakable(text):
        # text_split.py's own rule: a piece with no letter or digit (Unicode category L* or
        # N*) can't be spoken, so split_text drops it. Caught here, at the field stage, so a
        # text like "..." is 400 text_required before the reference-consistency checks below
        # ever run (FR-007's order) -- not left to surface only once splitting has silently
        # discarded the only piece there was.
        raise ApiError(400, "text_required", "text is required")

    # BC-09: a blank (or absent) instruction uses the default; a non-blank one is kept
    # exactly as given, not stripped. DECISION (final review): unlike `text`, instruction
    # keeps blank-before-length -- a whitespace-only instruction of any length still means
    # the default, not "too long" -- but control characters still run before the blank
    # check, same reason as `text` above.
    instruction_raw = _first(fields, "instruction")
    if instruction_raw is not None:
        _check_no_control_characters(instruction_raw, "instruction")  # BC-46
    if instruction_raw is None or not instruction_raw.strip():
        instruction = DEFAULT_INSTRUCTION
    else:
        if len(instruction_raw) > MAX_INSTRUCTION_CHARS:
            raise ApiError(
                400,
                "invalid_field",
                f"instruction must be at most {MAX_INSTRUCTION_CHARS:,} characters",
            )
        instruction = instruction_raw

    cfg_scale_raw = _first(fields, "cfg_scale")
    if cfg_scale_raw is None:
        cfg_scale = DEFAULT_CFG_SCALE
    else:
        cfg_scale = _parse_decimal(cfg_scale_raw, "cfg_scale")
        _check_decimal_range(
            cfg_scale_raw,
            cfg_scale,
            "cfg_scale",
            *CFG_SCALE_RANGE,
            low_inclusive=True,
            rule="finite and between 0 and 100",
        )

    seed_raw = _first(fields, "seed")
    if seed_raw is None:
        seed = DEFAULT_SEED
    else:
        seed_rule = "an integer between 0 and 4294967295"
        seed = _parse_int(seed_raw, "seed", seed_rule)
        _check_int_range(seed, "seed", *SEED_RANGE, seed_rule)

    temperature = _optional_decimal(
        fields, "temperature", *TEMPERATURE_RANGE, low_inclusive=False,
        rule="0, or greater than 0 and at most 10",
    )
    top_k = _optional_int(
        fields, "top_k", *TOP_K_RANGE, "0, or an integer between 1 and 10,000"
    )
    top_p = _optional_decimal(
        fields, "top_p", *TOP_P_RANGE, low_inclusive=False,
        rule="0, or greater than 0 and at most 1",
    )
    repetition_penalty = _optional_decimal(
        fields, "repetition_penalty", *REPETITION_PENALTY_RANGE, low_inclusive=True,
        rule="0, or between 0.0001 and 10",
    )
    max_new_tokens = _optional_int(
        fields,
        "max_new_tokens",
        *MAX_NEW_TOKENS_RANGE,
        f"0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING:,}",
    )

    split_chars_raw = _first(fields, "split_chars")
    if split_chars_raw is None:
        split_chars = settings.split_chars
    else:
        split_chars_rule = "an integer between 0 and 10,000"
        split_chars = _parse_int(split_chars_raw, "split_chars", split_chars_rule)
        _check_int_range(split_chars, "split_chars", *SPLIT_CHARS_RANGE, split_chars_rule)

    return SpeechRequest(
        text=text,
        instruction=instruction,
        reference=_build_reference(fields),
        cfg_scale=cfg_scale,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        split_chars=split_chars,
    )


@dataclass(frozen=True)
class VoiceCreateRequest:
    """`POST /v1/voices`'s fields, already validated (contracts/http-api.md "Fields").
    `name` is `None` for an unnamed voice."""

    audio_bytes: bytes
    ref_text: str
    name: str | None


def parse_voice_create(fields: Fields) -> VoiceCreateRequest:
    """Step 2 of `POST /v1/voices`'s order of checks.

    `ref_text` goes through the same rule as speech's (`_validated_text_field`: control
    characters and over-length are `400 invalid_field`; blank means absent). Then a missing
    `ref_audio` or `ref_text` is `400 voice_fields_required`, and only then is the name
    checked (`400 invalid_name`, also for the reserved `v_` prefix, BC-26), the order the
    contract lists them in. An attached but empty `ref_audio` part counts as present, as for
    speech: the decode step turns it into `400 invalid_audio`.
    """
    ref_text = _validated_text_field(fields, "ref_text", MAX_REF_TEXT_CHARS)
    if fields.ref_audio is None or ref_text is None:
        raise ApiError(400, "voice_fields_required", "ref_audio and ref_text are required")
    name = _first(fields, "name")
    if name is not None and not voice_file.is_valid_name(name):
        raise ApiError(
            400, "invalid_name", "name can only use letters, digits, dash and underscore"
        )
    return VoiceCreateRequest(audio_bytes=fields.ref_audio, ref_text=ref_text, name=name)
