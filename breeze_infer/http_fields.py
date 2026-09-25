"""`POST /v1/audio/speech` field parsing (data-model.md `SpeechRequest`, `ReferenceSpec`).

Happy path only (tasks.md T037): defaults, BC-02 (empty means absent), BC-09 (blank
instruction means the default), BC-10 (missing/blank text is `400 text_required`) and
FR-006's `0` -> `None` for the sampling fields. T048 completes this module: the strict
number grammar with ranges, duplicate-field detection, length limits, the control-character
rule, and the full `reference_conflict` / `ref_text_required` / `reference_required`
ordering. The structure here -- `Fields` keeping the form and the query string as separate
multi-dicts, `_first` as the one place that reads a named field -- is chosen so T048 can
slot its checks in without reshaping this module.

research.md R1: no `Form(...)` parameters -- FastAPI's silently keeps the last of a
duplicate field, and its parser limits can't be changed. research.md R6: a truncated
multipart body parses as an empty form with status 200, so `text` (and every other
required field) has to be checked explicitly here rather than relying on the parser to
reject a short body.

**T037 review 1** found that `request.form()` itself -- Starlette's own parser, not
FastAPI's `Form()` -- is also too permissive for this contract, in ways plain happy-path
testing didn't exercise. `read_fields` below no longer calls `request.form()` at all;
instead it drives the two content types by hand:

- `multipart/form-data` goes through `_StrictMultiPartParser`, a thin subclass of
  Starlette's own `MultiPartParser` (finding #4: its `on_part_end` silently re-decodes an
  invalid UTF-8 text part as latin-1 instead of rejecting it -- `_user_safe_decode` in
  `starlette/formparsers.py` -- and never checks the declared charset at all).
- `application/x-www-form-urlencoded` is parsed by hand with
  `urllib.parse.unquote_plus(..., errors="strict")` (finding #4 again: Starlette's own
  urlencoded parser calls `unquote_plus` with its default `errors="replace"`, which never
  raises either).

Every other field except `ref_audio` must be a string (finding #1); `ref_audio` must be a
file part, not a string, wherever it's given (finding #2). `ref_audio`'s bytes are read
bounded to `MAX_AUDIO_BYTES + 1` (finding #6), and every `UploadFile`/`FormData` this module
touches is closed before `read_fields` returns (finding #5) -- `Fields` only ever holds the
plain strings and bytes actually extracted, never Starlette's own form objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import unquote_plus

from fastapi import Request
from python_multipart.multipart import parse_options_header
from starlette.datastructures import FormData, QueryParams, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from breeze_infer.errors import ApiError
from breeze_infer.limits import MAX_AUDIO_BYTES, MAX_TEXT_CHARS
from breeze_infer.settings import Settings

# contracts/http-api.md "Fields": the defaults for POST /v1/audio/speech.
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
DEFAULT_CFG_SCALE = 1.0
DEFAULT_SEED = 42

# The limits passed to the multipart parser (research.md R6).
FORM_MAX_FILES = 1
FORM_MAX_FIELDS = 32
# T037 review 1, finding #3: a 4-byte UTF-8 code point (e.g. an emoji, or a CJK Extension-B
# character) percent-encodes to 12 ASCII bytes ("%XX" x 4), so a urlencoded `text` field at
# the full MAX_TEXT_CHARS (10,000) needs up to 120,000 bytes on the wire. The old flat
# 64 KiB limit -- copied uncritically from research.md's example -- would have cut that off
# as a generic parser error before the field-level length check (T048's `text_too_long`)
# ever got to run. `_read_urlencoded_pairs` enforces this same bound on each value by hand
# (Starlette's own per-field accounting no longer applies once its urlencoded parser is
# bypassed, per finding #4); `_StrictMultiPartParser` still gets it from Starlette's own
# per-part accounting, which was already value-only, so this only ever widens what fits.
FORM_MAX_PART_SIZE = MAX_TEXT_CHARS * 12


@dataclass(frozen=True)
class Fields:
    """The request's fields, already reduced to plain strings and bytes.

    Never Starlette's `FormData` (T037 review 1, finding #5): that type holds each
    `UploadFile`'s own open `SpooledTemporaryFile`, and `read_fields` closes every one of
    those before returning -- there would be nothing left downstream to read from even if
    something tried.

    `form` and `query` are both `QueryParams` (an immutable str -> str multi-dict; `getlist`
    gives every value for a name), kept apart rather than merged into one mapping so T048's
    duplicate check -- "`getlist(k)` has more than one value in the form or the query
    string, or the same key appears in both" (research.md R6) -- can be built directly from
    them instead of reconstructing which source each value came from.
    """

    form: QueryParams
    query: QueryParams
    ref_audio: bytes | None

    def values(self, name: str) -> list[str]:
        """Every value given for `name`: the form's values, then the query string's.

        This order is today's precedence, not a considered API: with no duplicate check yet
        (T048 adds `400 duplicate_field`), `_first` below just takes the first entry, so a
        field given in both places silently resolves to its form value.
        """
        return [*self.form.getlist(name), *self.query.getlist(name)]


async def read_fields(request: Request) -> Fields:
    """Read the body (bounded per research.md R6) and merge it with the query string.

    Routes on the request's declared media type (finding #7): `multipart/form-data` and
    `application/x-www-form-urlencoded` are parsed as the module docstring describes;
    anything else is `400 invalid_field` *unless* the body is empty -- a query-only request
    with no body has no content type worth trusting, and still has to work.
    """
    if "ref_audio" in request.query_params:
        # ref_audio can only ever be a file part, and a query string can't carry one
        # (finding #2's other half; the multipart/urlencoded halves are below).
        raise ApiError(400, "invalid_field", "ref_audio must be a file part")

    media_type = _media_type(request)
    if media_type == "multipart/form-data":
        pairs, ref_audio = await _read_multipart_fields(request)
    elif media_type == "application/x-www-form-urlencoded":
        pairs = await _read_urlencoded_pairs(request)
        _reject_ref_audio_text(pairs)
        ref_audio = None
    else:
        body = await request.body()  # already bounded by BodyLimitMiddleware
        if body:
            raise ApiError(400, "invalid_field", "unsupported content type")
        pairs, ref_audio = [], None

    return Fields(form=QueryParams(pairs), query=request.query_params, ref_audio=ref_audio)


def _media_type(request: Request) -> str:
    """The request's Content-Type media type, lowercased and without parameters; `""` when
    the header is absent."""
    header = request.headers.get("content-type", "")
    media_type, _ = parse_options_header(header)
    return media_type.decode("latin-1").lower()


def _reject_ref_audio_text(pairs: list[tuple[str, str]]) -> None:
    """Finding #2, the urlencoded half: `ref_audio` can only be a multipart file part, so a
    same-named urlencoded field is rejected the same way a string multipart field is
    (`_read_ref_audio_part` below)."""
    if any(name == "ref_audio" for name, _ in pairs):
        raise ApiError(400, "invalid_field", "ref_audio must be a file part")


async def _read_urlencoded_pairs(request: Request) -> list[tuple[str, str]]:
    """Hand-parsed `application/x-www-form-urlencoded`, `errors="strict"` (finding #4).

    `request.body()` is already bounded by `BodyLimitMiddleware` overall (research.md R6),
    so this only needs its own per-value bound (`FORM_MAX_PART_SIZE`, finding #3). The body
    is decoded latin-1 first -- a lossless, always-successful byte<->codepoint mapping, used
    purely so the ASCII structure (`&`, `=`) can be split on with plain `str` methods; the
    real decode is each `unquote_plus` call's own `encoding="utf-8"`, which is where a
    genuinely invalid UTF-8 payload (or one that was never percent-escaped UTF-8 to begin
    with) is rejected.
    """
    body = await request.body()
    if not body:
        return []

    pairs: list[tuple[str, str]] = []
    for chunk in body.decode("latin-1").split("&"):
        if not chunk:
            continue
        raw_name, _, raw_value = chunk.partition("=")
        if len(raw_value) > FORM_MAX_PART_SIZE:
            # Mirrors `MultiPartParser.on_part_data`'s own check, which is value-only too
            # (headers/field names aren't counted against `max_part_size` there either).
            raise ApiError(400, "invalid_field", "could not parse the request body")
        name = _unquote_utf8(raw_name, field=raw_name)
        value = _unquote_utf8(raw_value, field=name)
        pairs.append((name, value))
    return pairs


def _unquote_utf8(raw: str, *, field: str) -> str:
    try:
        return unquote_plus(raw, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ApiError(400, "invalid_field", f"{field} must be UTF-8 text") from None


class _StrictMultiPartParser(MultiPartParser):
    """Starlette's own `MultiPartParser.on_part_end` (`starlette/formparsers.py`) decodes a
    text part with `_user_safe_decode`, which silently falls back to a latin-1 decode when
    the part isn't valid under its charset -- so a genuinely corrupt UTF-8 part is never
    rejected, just mojibake'd (finding #4). This overrides only that one callback: the
    declared charset must be `utf-8` (this contract has never supported anything else), and
    the bytes must actually decode as `utf-8` with `errors="strict"`, or the field is
    rejected the same way every other field-level error in this module is. Everything else
    -- boundary parsing, file spooling, the file/field count and size limits -- is untouched
    Starlette machinery, reached via `super()`.
    """

    def on_part_end(self) -> None:
        if self._current_part.file is not None:
            super().on_part_end()
            return
        field_name = self._current_part.field_name
        if self._charset.lower() != "utf-8":
            raise ApiError(400, "invalid_field", f"{field_name} must be UTF-8 text")
        try:
            value = self._current_part.data.decode("utf-8")
        except UnicodeDecodeError:
            raise ApiError(400, "invalid_field", f"{field_name} must be UTF-8 text") from None
        self.items.append((field_name, value))


async def _read_multipart_fields(request: Request) -> tuple[list[tuple[str, str]], bytes | None]:
    """`multipart/form-data`, via `_StrictMultiPartParser` -- not `request.form()`, which
    constructs the base (silently-lossy) parser and gives no way to swap it in (finding #4).

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
        pairs = _text_pairs(form)
        ref_audio = await _read_ref_audio_part(form.get("ref_audio"))
    finally:
        # Finding #5: close every `UploadFile` this form holds -- `ref_audio`'s, and (on the
        # finding #1 path inside `_text_pairs`) any file wrongly sent under another field's
        # name -- before returning. `Fields` never holds `form` itself.
        await form.close()

    return pairs, ref_audio


def _text_pairs(form: FormData) -> list[tuple[str, str]]:
    """Every field except `ref_audio`, checked to actually be text (finding #1): a file part
    sent under any other field name is rejected outright."""
    pairs: list[tuple[str, str]] = []
    for name, value in form.multi_items():
        if name == "ref_audio":
            continue
        if isinstance(value, UploadFile):
            raise ApiError(400, "invalid_field", f"{name} must be a text field")
        pairs.append((name, value))
    return pairs


async def _read_ref_audio_part(part: str | UploadFile | None) -> bytes | None:
    """`form.get` plus an `isinstance` check (finding #8): a string `ref_audio` is rejected
    (finding #2), and a real file part is read bounded to one byte past `MAX_AUDIO_BYTES`
    (finding #6) -- enough for `reference_audio.decode()` to tell "too big" from "right at
    the limit" later without this module ever buffering more than that itself.
    """
    if part is None:
        return None
    if not isinstance(part, UploadFile):
        raise ApiError(400, "invalid_field", "ref_audio must be a file part")
    return await part.read(MAX_AUDIO_BYTES + 1)


def _first(fields: Fields, name: str) -> str | None:
    """The field's first value, or `None` when it's absent or empty (BC-02).

    Happy-path lookup: it doesn't check for a second value. T048 replaces this call site
    with a duplicate check that raises `400 duplicate_field` before falling back to this
    same "first (and only) value, empty means absent" reading.
    """
    values = fields.values(name)
    if not values or values[0] == "":
        return None
    return values[0]


# A parser matching _parse_int/_parse_float's own shape, used by _optional_number so its
# return type tracks whichever one is passed in (finding #9) -- `int | None` for `_parse_int`,
# `float | None` for `_parse_float` -- rather than the wider `int | float | None` either call
# site would otherwise have to narrow back down itself.
_Number = TypeVar("_Number", int, float)


# Lenient on purpose, for now: Python's `int`/`float` accept a wider grammar than the
# contract does (leading `+`, surrounding whitespace, `inf`/`nan`, underscore digit
# separators), and don't enforce the contract's ranges at all. That's fine for the happy
# path this task covers; T048 swaps these two functions for the strict regex grammar and
# range checks in contracts/http-api.md, without changing any other call site.
def _parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ApiError(400, "invalid_field", f"{field} must be an integer") from None


def _parse_float(value: str, field: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ApiError(400, "invalid_field", f"{field} must be a number") from None


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
    ref_text = _first(fields, "ref_text")

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


def _optional_number(
    fields: Fields, name: str, parser: Callable[[str, str], _Number]
) -> _Number | None:
    """FR-006: `0` or absent means the model default (`None`) for a sampling field."""
    raw = _first(fields, name)
    if raw is None:
        return None
    value = parser(raw, name)
    return None if value == 0 else value


def parse_speech(fields: Fields, settings: Settings) -> SpeechRequest:
    """data-model.md "SpeechRequest": defaults, BC-02/BC-09/BC-10 and FR-006 applied."""
    text = _first(fields, "text")
    if text is None or not text.strip():
        raise ApiError(400, "text_required", "text is required")

    # BC-09: a blank (or absent) instruction uses the default; a non-blank one is kept
    # exactly as given, not stripped.
    instruction_raw = _first(fields, "instruction")
    instruction = (
        DEFAULT_INSTRUCTION
        if instruction_raw is None or not instruction_raw.strip()
        else instruction_raw
    )

    cfg_scale_raw = _first(fields, "cfg_scale")
    cfg_scale = (
        _parse_float(cfg_scale_raw, "cfg_scale")
        if cfg_scale_raw is not None
        else DEFAULT_CFG_SCALE
    )

    seed_raw = _first(fields, "seed")
    seed = _parse_int(seed_raw, "seed") if seed_raw is not None else DEFAULT_SEED

    temperature = _optional_number(fields, "temperature", _parse_float)
    top_k = _optional_number(fields, "top_k", _parse_int)
    top_p = _optional_number(fields, "top_p", _parse_float)
    repetition_penalty = _optional_number(fields, "repetition_penalty", _parse_float)
    max_new_tokens = _optional_number(fields, "max_new_tokens", _parse_int)

    split_chars_raw = _first(fields, "split_chars")
    split_chars = (
        _parse_int(split_chars_raw, "split_chars")
        if split_chars_raw is not None
        else settings.split_chars
    )

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
