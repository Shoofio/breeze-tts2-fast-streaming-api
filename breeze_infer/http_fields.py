"""`POST /v1/audio/speech` field parsing (data-model.md `SpeechRequest`, `ReferenceSpec`).

tasks.md T037 built the happy path: defaults, BC-02 (empty means absent), BC-09 (blank
instruction means the default), BC-10 (missing/blank text is `400 text_required`) and
FR-006's `0` -> `None` for the sampling fields. T048 (contracts/http-api.md, data-model.md
`SpeechRequest`) completes it: the strict ASCII-only number grammar with ranges (T048's
`_INT_LITERAL`/`_DECIMAL_LITERAL` below), duplicate-field detection (BC-08, `_first`),
length limits and the control-character rule (BC-05, BC-46), and the full
`reference_conflict` / `ref_text_required` / `reference_required` ordering (`_build_reference`).
`Fields` keeps the form and the query string as separate multi-dicts specifically so
`_first` can tell "given twice in the form", "given twice in the query" and "given in both"
apart for `duplicate_field`'s own bookkeeping, without either check needing to know which
source a value came from.

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
import math
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qsl

from fastapi import Request
from python_multipart.multipart import parse_options_header
from starlette.datastructures import FormData, QueryParams, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from breeze_infer.errors import ApiError
from breeze_infer.limits import (
    MAX_AUDIO_BYTES,
    MAX_INSTRUCTION_CHARS,
    MAX_NEW_TOKENS_CEILING,
    MAX_REF_TEXT_CHARS,
    MAX_TEXT_CHARS,
)
from breeze_infer.settings import Settings

# contracts/http-api.md "Fields": the defaults for POST /v1/audio/speech.
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
DEFAULT_CFG_SCALE = 1.0
DEFAULT_SEED = 42

# The limits passed to the multipart parser, and to _urlencoded_pairs_sync by hand
# (research.md R6).
FORM_MAX_FILES = 1
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

    `form` and `query` are kept apart rather than merged into one mapping so the duplicate
    check -- "`getlist(k)` has more than one value in the form or the query string, or the
    same key appears in both" (research.md R6, BC-08) -- can be built directly from them
    instead of reconstructing which source each value came from.
    """

    form: QueryParams
    query: QueryParams
    ref_audio: bytes | None


async def read_fields(request: Request) -> Fields:
    """Read the body (bounded per research.md R6) and merge it with the query string.

    The query string is parsed by the same strict routine as a urlencoded body (review 2
    finding #5), from `request.scope["query_string"]` -- the raw wire bytes, not
    `request.query_params` (Starlette's own lenient parse).

    Routes on the request's declared media type: `multipart/form-data` and
    `application/x-www-form-urlencoded` are parsed as the module docstring describes;
    anything else is `400 invalid_field` *unless* the body is empty -- a query-only request
    with no body has no content type worth trusting, and still has to work (review 1
    finding #7, review 2 finding #8).
    """
    query_pairs = await _parse_urlencoded_bytes(request.scope["query_string"])
    _reject_ref_audio_text(query_pairs)

    media_type, charset = _content_type(request)
    if media_type == "multipart/form-data":
        pairs, ref_audio = await _read_multipart_fields(request)
    elif media_type == "application/x-www-form-urlencoded":
        if not _is_utf8_charset(charset):
            raise ApiError(400, "invalid_field", "request body must be UTF-8 text")
        body = await request.body()  # already bounded by BodyLimitMiddleware
        pairs = await _parse_urlencoded_bytes(body)
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


async def _parse_urlencoded_bytes(data: bytes) -> list[tuple[str, str]]:
    """`data` (a request body or `scope["query_string"]`), parsed and strictly UTF-8
    decoded. Runs inline for a small payload, or off the event loop for a large one
    (review 2 finding #9) -- `_urlencoded_pairs_sync` is plain CPU-bound work either way.
    """
    if len(data) > _TO_THREAD_THRESHOLD:
        return await asyncio.to_thread(_urlencoded_pairs_sync, data)
    return _urlencoded_pairs_sync(data)


def _urlencoded_pairs_sync(body: bytes) -> list[tuple[str, str]]:
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
        # huge body with many tiny fields is rejected in roughly one pass over the bytes.
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


async def _read_multipart_fields(request: Request) -> tuple[list[tuple[str, str]], bytes | None]:
    """`multipart/form-data`, via `_StrictMultiPartParser` -- not `request.form()`, which
    constructs the base (silently-lossy) parser and gives no way to swap it in.

    Mirrors what `Request._get_form` does for this content type (`starlette/requests.py`):
    the same limits, and the same fallback for a boundary/limit error (`MultiPartException`,
    Starlette's own class in `formparsers.py` -- unrelated to
    `python_multipart.exceptions.FormParserError`, which `errors.py` already maps to `400`
    for the lower-level parser's own malformed-syntax errors, which still propagate
    unchanged since this module never catches them).
    """
    parser = _StrictMultiPartParser(
        request.headers,
        request.stream(),
        max_files=FORM_MAX_FILES,
        max_fields=FORM_MAX_FIELDS,
        max_part_size=FORM_MAX_PART_SIZE,
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise ApiError(400, "invalid_field", "could not parse the request body") from exc

    try:
        pairs, ref_audio_part = _split_multipart_fields(form)
        ref_audio = await _read_ref_audio_bytes(ref_audio_part)
    finally:
        # Every UploadFile this form holds -- ref_audio's, and any file wrongly sent under
        # another field's name -- is closed before returning. `Fields` never holds `form`.
        await form.close()

    return pairs, ref_audio


def _split_multipart_fields(form: FormData) -> tuple[list[tuple[str, str]], UploadFile | None]:
    """One pass over every part, in wire order (review 2 finding #3): every `ref_audio`
    entry must be a file, checked as it's encountered, not just the *last* one --
    `form.get("ref_audio")` alone would miss an earlier, invalid text `ref_audio` sent
    before a later, valid file part under the same name. Every other field must be text
    (review 1 finding #1): a file part sent under any other field name is rejected outright.
    """
    pairs: list[tuple[str, str]] = []
    ref_audio_part: UploadFile | None = None
    for name, value in form.multi_items():
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


def _first(fields: Fields, name: str) -> str | None:
    """The field's value, or `None` when it's absent or empty (BC-02).

    BC-08: a field given more than once -- twice in the form, twice in the query string, or
    once in each -- is `400 duplicate_field` before anything else is checked about it.
    Reading `form.getlist`/`query.getlist` separately (rather than a single merged list) is
    what lets this tell those cases apart, which the message needs in order to stay
    accurate regardless of which source doubled up.
    """
    form_values = fields.form.getlist(name)
    query_values = fields.query.getlist(name)
    if len(form_values) > 1 or len(query_values) > 1 or (form_values and query_values):
        raise ApiError(400, "duplicate_field", f"{_label(name)} was given more than once")
    values = [*form_values, *query_values]
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
_INT_LITERAL = re.compile(r"[+-]?\d+", re.ASCII)
_DECIMAL_LITERAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def _parse_int(value: str, field: str) -> int:
    """The strict integer grammar (BC-01): reject anything `int()` itself is more lenient
    about than the contract is -- surrounding whitespace, a non-ASCII digit, underscore
    digit separators -- before ever calling `int()`, so only a string already known to
    match `^[+-]?\\d+$` reaches it."""
    if _INT_LITERAL.fullmatch(value) is None:
        raise ApiError(400, "invalid_field", f"{field} must be an integer")
    return int(value)


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


def _is_zero_literal(value: str) -> bool:
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


def _check_int_range(value: int, field: str, low: int, high: int, rule: str) -> None:
    if not (low <= value <= high):
        raise ApiError(400, "invalid_field", f"{field} must be {rule}")


def _check_decimal_range(
    value: float, field: str, low: float, high: float, *, low_inclusive: bool, rule: str
) -> None:
    """`math.isfinite` guards an overflowed literal (e.g. `cfg_scale=1e400`, which parses to
    `inf`): every field this module validates has a finite `high`, so `inf` would otherwise
    fail the ordinary `value <= high` comparison anyway, but checking it explicitly doesn't
    depend on that staying true for every future caller."""
    in_range = (value >= low) if low_inclusive else (value > low)
    if not (math.isfinite(value) and in_range and value <= high):
        raise ApiError(400, "invalid_field", f"{field} must be {rule}")


def _optional_int(fields: Fields, name: str, low: int, high: int, rule: str) -> int | None:
    """FR-006: `0` (or an absent field) means the model default (`None`); otherwise the
    value must fall in `[low, high]`. Python's `int` has arbitrary precision -- no overflow
    or underflow -- so unlike `_optional_decimal` below, the sentinel is decided from the
    parsed value directly."""
    raw = _first(fields, name)
    if raw is None:
        return None
    value = _parse_int(raw, name)
    if value == 0:
        return None
    _check_int_range(value, name, low, high, rule)
    return value


def _optional_decimal(
    fields: Fields, name: str, low: float, high: float, *, low_inclusive: bool, rule: str
) -> float | None:
    """FR-006: `0` (or an absent field) means the model default (`None`); otherwise the
    value must fall in the given range. The sentinel is decided from the literal text
    (`_is_zero_literal`), not the parsed `float`, per the review finding `_is_zero_literal`
    documents -- so a nonzero literal that underflows still reaches, and fails,
    `_check_decimal_range` instead of being mistaken for the default."""
    raw = _first(fields, name)
    if raw is None:
        return None
    value = _parse_decimal(raw, name)
    if _is_zero_literal(raw):
        return None
    _check_decimal_range(value, name, low, high, low_inclusive=low_inclusive, rule=rule)
    return value


def _check_no_control_characters(value: str, field: str) -> None:
    """BC-46: reject any Unicode general-category `Cc` (control) character except tab, CR
    and LF. `Cc` is precisely C0 (`\\x00`-`\\x1f`, e.g. NUL and ESC), DEL (`\\x7f`) and the
    C1 controls (`\\x80`-`\\x9f`, e.g. NEL `\\x85`) -- exactly the set the contract means by
    "control characters", so this is checked via `unicodedata.category` rather than a fixed
    codepoint list.
    """
    for ch in value:
        if ch not in "\t\r\n" and unicodedata.category(ch) == "Cc":
            raise ApiError(
                400, "invalid_field", f"{field} must not contain control characters"
            )


def _validated_text_field(fields: Fields, name: str, max_chars: int) -> str | None:
    """A text field's value (`None` when absent, BC-02), length- and control-character-
    checked (BC-05/BC-46) -- shared by `instruction` and `ref_text`, whose only difference
    from each other is their max length and (for `instruction`) the blank-means-default
    rule the caller applies around this. `text` doesn't use this: it has its own
    `text_required`/`text_too_long` codes instead of `invalid_field`.
    """
    raw = _first(fields, name)
    if raw is None:
        return None
    if len(raw) > max_chars:
        raise ApiError(400, "invalid_field", f"{name} must be at most {max_chars} characters")
    _check_no_control_characters(raw, name)
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
    """
    text = _first(fields, "text")
    if text is None or not text.strip():
        raise ApiError(400, "text_required", "text is required")  # BC-10
    if len(text) > MAX_TEXT_CHARS:  # BC-05
        raise ApiError(400, "text_too_long", "text is too long")
    _check_no_control_characters(text, "text")  # BC-46

    # BC-09: a blank (or absent) instruction uses the default; a non-blank one is kept
    # exactly as given, not stripped -- its length and control characters are only
    # checked once it's known not to be blank, since a huge whitespace-only string is
    # still "blank" (BC-09), not "too long".
    instruction_raw = _first(fields, "instruction")
    if instruction_raw is None or not instruction_raw.strip():
        instruction = DEFAULT_INSTRUCTION
    else:
        if len(instruction_raw) > MAX_INSTRUCTION_CHARS:
            raise ApiError(
                400,
                "invalid_field",
                f"instruction must be at most {MAX_INSTRUCTION_CHARS} characters",
            )
        _check_no_control_characters(instruction_raw, "instruction")
        instruction = instruction_raw

    cfg_scale_raw = _first(fields, "cfg_scale")
    if cfg_scale_raw is None:
        cfg_scale = DEFAULT_CFG_SCALE
    else:
        cfg_scale = _parse_decimal(cfg_scale_raw, "cfg_scale")
        _check_decimal_range(
            cfg_scale,
            "cfg_scale",
            0,
            100,
            low_inclusive=True,
            rule="finite and between 0 and 100",
        )

    seed_raw = _first(fields, "seed")
    if seed_raw is None:
        seed = DEFAULT_SEED
    else:
        seed = _parse_int(seed_raw, "seed")
        _check_int_range(seed, "seed", 0, 4_294_967_295, "an integer between 0 and 4294967295")

    temperature = _optional_decimal(
        fields, "temperature", 0, 10, low_inclusive=False,
        rule="0, or greater than 0 and at most 10",
    )
    top_k = _optional_int(fields, "top_k", 1, 10_000, "0, or an integer between 1 and 10000")
    top_p = _optional_decimal(
        fields, "top_p", 0, 1, low_inclusive=False, rule="0, or greater than 0 and at most 1"
    )
    repetition_penalty = _optional_decimal(
        fields, "repetition_penalty", 0.0001, 10, low_inclusive=True,
        rule="0, or between 0.0001 and 10",
    )
    max_new_tokens = _optional_int(
        fields,
        "max_new_tokens",
        1,
        MAX_NEW_TOKENS_CEILING,
        f"0, or an integer between 1 and {MAX_NEW_TOKENS_CEILING}",
    )

    split_chars_raw = _first(fields, "split_chars")
    if split_chars_raw is None:
        split_chars = settings.split_chars
    else:
        split_chars = _parse_int(split_chars_raw, "split_chars")
        _check_int_range(split_chars, "split_chars", 0, 10_000, "an integer between 0 and 10000")

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
