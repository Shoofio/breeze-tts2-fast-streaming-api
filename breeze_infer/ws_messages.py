"""tasks.md T075: parse a WebSocket client text frame (contracts/ws-api.md "Client ->
server") into a typed message or a `WsError`. Pure -- no I/O, no `Settings`, no session
state (that belongs to `ws_session.py`/`ws_server.py`, T076/T077).

Numbers go through `json.loads(parse_int=_JsonInt, parse_float=_JsonFloat)` so each field
check keeps the exact source literal, not just the rounded `int`/`float` -- that's what
lets this module reuse http_fields.py's exact zero-sentinel (FR-006) and boundary checks
(`is_zero_literal`, `decimal_literal_in_range`) instead of re-deriving them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

from breeze_infer.http_fields import (
    CFG_SCALE_RANGE,
    MAX_INT_LITERAL_DIGITS,
    MAX_NEW_TOKENS_RANGE,
    REPETITION_PENALTY_RANGE,
    SEED_RANGE,
    SPLIT_CHARS_RANGE,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
    TOP_P_RANGE,
    TextFieldError,
    decimal_literal_in_range,
    is_valid_voice_id,
    is_zero_literal,
    significant_int_digits,
    validate_bounded_text,
)
from breeze_infer.limits import MAX_INSTRUCTION_CHARS, MAX_REF_TEXT_CHARS
from breeze_infer.text_rules import has_control_characters, is_utf8_encodable


@dataclass(frozen=True)
class Start:
    """contracts/ws-api.md "Client -> server", `start` row. A field is `None` when the
    client left it unset -- absent, `null`, or (BC-02, `start` only) an empty string;
    `ws_session` applies each field's own default.
    """

    TYPE: ClassVar[str] = "start"

    voice_id: str | None
    instruction: str | None
    ref_text: str | None
    cfg_scale: float | None
    seed: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    max_new_tokens: int | None
    split_chars: int | None


@dataclass(frozen=True)
class Text:
    """`text` is required -- unlike `flush`/`end`, never marked "(optional)"."""

    TYPE: ClassVar[str] = "text"

    text: str


@dataclass(frozen=True)
class Flush:
    TYPE: ClassVar[str] = "flush"

    text: str | None


@dataclass(frozen=True)
class End:
    TYPE: ClassVar[str] = "end"

    text: str | None


@dataclass(frozen=True)
class Instruction:
    TYPE: ClassVar[str] = "instruction"

    instruction: str


@dataclass(frozen=True)
class Cancel:
    """No fields."""

    TYPE: ClassVar[str] = "cancel"


Message = Start | Text | Flush | End | Instruction | Cancel


@dataclass(frozen=True)
class WsError:
    """contracts/ws-api.md's `error` event, minus the wire `type` (`ws_server.py` adds
    that). `request_type` is the client's `type` when it was a readable string -- true
    of every field-level error inside a recognized message, not just `unknown_type` --
    and `None` only for `invalid_json` and a missing/non-string `type` itself.
    """

    code: str
    message: str
    request_type: str | None


class _Invalid(Exception):
    """Unwinds a `start`/`text`/`flush`/`end`/`instruction` parse on the first invalid
    field -- mirrors http_fields.py's own use of `ApiError` for the same control flow.
    """

    def __init__(self, error: WsError) -> None:
        super().__init__(error.message)
        self.error = error


class _JsonInt(str):
    """A JSON integer token (no `.`/`e`/`E`), carrying its exact source digits as a
    `str` subclass -- never produced for an actual JSON string, so `isinstance(x,
    _JsonInt)` means "the client sent a JSON integer", not one that merely looks like it.
    """


class _JsonFloat(str):
    """Same as `_JsonInt`, for a JSON decimal/exponent token."""


def _is_json_string(value: object) -> bool:
    """`type(value) is str`, not `isinstance`: excludes `_JsonInt`/`_JsonFloat` (both
    `str` subclasses), so a JSON number is never mistaken for a string field's value.
    """
    return type(value) is str


def _blank_means_absent(value: str | None) -> str | None:
    """BC-02, extended to `start` (contracts/ws-api.md): an empty string counts as
    absent, same as an unset field, on HTTP and now here."""
    return None if value == "" else value


def _string_field(data: dict[str, object], name: str, *, request_type: str) -> str | None:
    """A JSON string, or `None` if absent/`null`; anything else is `invalid_field`. Used
    for `voice_id`, which BC-46 was never extended to (`_text_field` below is)."""
    value = data.get(name)
    if value is None:
        return None
    if not _is_json_string(value):
        raise _Invalid(WsError("invalid_field", f"{name} must be a string", request_type))
    return value


def _text_field(data: dict[str, object], name: str, *, request_type: str) -> str | None:
    """`_string_field`, plus BC-46 (a disallowed control character is `invalid_field`)
    and a lone-surrogate check: a `\\uD800`-style escape decodes to a real Python `str`
    that can never be encoded back to UTF-8 (`text_rules.is_utf8_encodable`, the same
    rule `voice_file` applies to a saved `ref_text`). Used for `text`, `start`'s
    `instruction`/`ref_text`, and the standalone `instruction` message.
    """
    value = _string_field(data, name, request_type=request_type)
    if value is None:
        return None
    if has_control_characters(value):
        raise _Invalid(
            WsError("invalid_field", f"{name} must be free of control characters", request_type)
        )
    if not is_utf8_encodable(value):
        raise _Invalid(WsError("invalid_field", f"{name} must be valid UTF-8 text", request_type))
    return value


def _decimal_field(
    data: dict[str, object],
    name: str,
    low: str,
    high: str,
    *,
    low_inclusive: bool,
    zero_means_default: bool,
    request_type: str,
) -> float | None:
    """A `start` decimal field (`cfg_scale`, `temperature`, `top_p`,
    `repetition_penalty`), sharing http_fields' range. `zero_means_default` (FR-006) is
    decided from the literal's digits (`is_zero_literal`), not the parsed float, so a
    literal like `1e-400` that underflows to `0.0` isn't mistaken for a true `0`.
    """
    value = data.get(name)
    if value is None or value == "":
        return None  # BC-02: an empty string counts as absent, same as every other field
    if not isinstance(value, (_JsonInt, _JsonFloat)):
        raise _Invalid(WsError("invalid_field", f"{name} must be a number", request_type))
    literal = str(value)
    if zero_means_default and is_zero_literal(literal):
        return None
    parsed = float(literal)
    if not decimal_literal_in_range(literal, parsed, low, high, low_inclusive=low_inclusive):
        raise _Invalid(WsError("invalid_field", f"{name} is out of range", request_type))
    return parsed


def _int_field(
    data: dict[str, object],
    name: str,
    low: int,
    high: int,
    *,
    zero_means_default: bool,
    request_type: str,
) -> int | None:
    """A `start` integer field (`seed`, `top_k`, `max_new_tokens`, `split_chars`),
    sharing http_fields' range. Only a `_JsonInt` is accepted -- a `_JsonFloat` like
    `5.0` is `invalid_field` rather than silently truncated, same as an integer field's
    grammar rejects a decimal-shaped string on HTTP.
    """
    value = data.get(name)
    if value is None or value == "":
        return None  # BC-02: an empty string counts as absent, same as every other field
    if not isinstance(value, _JsonInt):
        raise _Invalid(WsError("invalid_field", f"{name} must be an integer", request_type))
    sign, significant_digits = significant_int_digits(str(value))
    if len(significant_digits) > MAX_INT_LITERAL_DIGITS:
        raise _Invalid(WsError("invalid_field", f"{name} is out of range", request_type))
    parsed = int(sign + significant_digits)
    if zero_means_default and parsed == 0:
        return None
    if not (low <= parsed <= high):
        raise _Invalid(WsError("invalid_field", f"{name} is out of range", request_type))
    return parsed


def _bounded_text(
    raw: str | None, max_chars: int, *, name: str, request_type: str
) -> str | None:
    """`http_fields.validate_bounded_text` (control characters, then UTF-8, then blank,
    then length -- shared with `ref_text`'s own HTTP validation, review 46/47), raising
    this module's `_Invalid` for whichever reason it returns instead of a bare value.
    `raw` is already known to be a JSON string or `None` (`_string_field`'s job).
    """
    result = validate_bounded_text(raw, max_chars, check_utf8=True)
    if result is TextFieldError.CONTROL_CHARACTERS:
        raise _Invalid(
            WsError("invalid_field", f"{name} must be free of control characters", request_type)
        )
    if result is TextFieldError.NOT_UTF8:
        raise _Invalid(WsError("invalid_field", f"{name} must be valid UTF-8 text", request_type))
    if result is TextFieldError.TOO_LONG:
        raise _Invalid(
            WsError(
                "invalid_field", f"{name} must be at most {max_chars:,} characters", request_type
            )
        )
    return result


def _parse_start(data: dict[str, object]) -> Start | WsError:
    try:
        voice_id = _blank_means_absent(_string_field(data, "voice_id", request_type="start"))
        if voice_id is not None and not is_valid_voice_id(voice_id):
            raise _Invalid(
                WsError("invalid_field", "voice_id must be a voice name or v_ id", "start")
            )
        instruction = _bounded_text(
            _string_field(data, "instruction", request_type="start"),
            MAX_INSTRUCTION_CHARS,
            name="instruction",
            request_type="start",
        )
        ref_text = _bounded_text(
            _string_field(data, "ref_text", request_type="start"),
            MAX_REF_TEXT_CHARS,
            name="ref_text",
            request_type="start",
        )
        cfg_scale = _decimal_field(
            data, "cfg_scale", *CFG_SCALE_RANGE,
            low_inclusive=True, zero_means_default=False, request_type="start",
        )
        seed = _int_field(
            data, "seed", *SEED_RANGE, zero_means_default=False, request_type="start"
        )
        temperature = _decimal_field(
            data, "temperature", *TEMPERATURE_RANGE,
            low_inclusive=False, zero_means_default=True, request_type="start",
        )
        top_k = _int_field(
            data, "top_k", *TOP_K_RANGE, zero_means_default=True, request_type="start"
        )
        top_p = _decimal_field(
            data, "top_p", *TOP_P_RANGE,
            low_inclusive=False, zero_means_default=True, request_type="start",
        )
        repetition_penalty = _decimal_field(
            data, "repetition_penalty", *REPETITION_PENALTY_RANGE,
            low_inclusive=True, zero_means_default=True, request_type="start",
        )
        max_new_tokens = _int_field(
            data, "max_new_tokens", *MAX_NEW_TOKENS_RANGE,
            zero_means_default=True, request_type="start",
        )
        split_chars = _int_field(
            data, "split_chars", *SPLIT_CHARS_RANGE,
            zero_means_default=False, request_type="start",
        )
    except _Invalid as exc:
        return exc.error

    if ref_text is not None and voice_id is None:
        # BC-13: `ref_text` without `voice_id` -> reference_required, previous session
        # unchanged. No exact message is specified for this row (unlike unknown_voice's).
        return WsError("reference_required", "ref_text needs voice_id", "start")

    return Start(
        voice_id=voice_id,
        instruction=instruction,
        ref_text=ref_text,
        cfg_scale=cfg_scale,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        split_chars=split_chars,
    )


def _parse_text(data: dict[str, object]) -> Text | WsError:
    try:
        text = _text_field(data, "text", request_type="text")
    except _Invalid as exc:
        return exc.error
    if text is None:
        # Unlike flush/end, the contract never marks `text`'s `text` "(optional)".
        return WsError("invalid_field", "text is required", "text")
    return Text(text=text)


def _parse_flush(data: dict[str, object]) -> Flush | WsError:
    try:
        text = _text_field(data, "text", request_type="flush")
    except _Invalid as exc:
        return exc.error
    return Flush(text=text)


def _parse_end(data: dict[str, object]) -> End | WsError:
    try:
        text = _text_field(data, "text", request_type="end")
    except _Invalid as exc:
        return exc.error
    return End(text=text)


def _parse_instruction(data: dict[str, object]) -> Instruction | WsError:
    try:
        raw = _string_field(data, "instruction", request_type="instruction")
        if raw is None:
            return WsError("invalid_field", "instruction is required", "instruction")
        # `_bounded_text`'s result is discarded, not kept: a blank `raw` must stay valid
        # as itself (BC-37 lives in `ws_session`, not here) rather than collapse to
        # `None` the way `start`'s `instruction` does -- this call is only for its
        # control-character/UTF-8/length side effect, raising `_Invalid` on failure.
        _bounded_text(raw, MAX_INSTRUCTION_CHARS, name="instruction", request_type="instruction")
    except _Invalid as exc:
        return exc.error
    return Instruction(instruction=raw)


_PARSERS = {
    Start.TYPE: _parse_start,
    Text.TYPE: _parse_text,
    Flush.TYPE: _parse_flush,
    End.TYPE: _parse_end,
    Instruction.TYPE: _parse_instruction,
}


def _reject_non_finite_constant(token: str) -> float:
    """`json.loads`'s `parse_constant` hook: Python otherwise accepts the bare tokens
    `NaN`/`Infinity`/`-Infinity` anywhere a number goes, a non-standard extension to
    RFC 8259 that a "real JSON parser" (contracts/ws-api.md) shouldn't accept either.
    """
    raise ValueError(f"{token} is not valid JSON")


def parse(raw: str) -> Message | WsError:
    """contracts/ws-api.md "Client -> server": one WebSocket text frame -> a typed
    message or a `WsError`. See the module docstring for the `parse_int`/`parse_float`
    hooks this relies on to keep each number's exact literal text.
    """
    try:
        data = json.loads(
            raw,
            parse_int=_JsonInt,
            parse_float=_JsonFloat,
            parse_constant=_reject_non_finite_constant,
        )
    except (ValueError, RecursionError):
        # ValueError covers both json.JSONDecodeError (a subclass) and our own
        # parse_constant rejection above. RecursionError is json.loads's own recursive
        # descent parser hitting Python's stack limit on deeply nested input (an array
        # or object nested hundreds of thousands deep) -- uncaught, it would escape this
        # function and crash whatever's reading the socket; contracts/ws-api.md promises
        # the connection stays open after any malformed message, this one included.
        return WsError("invalid_json", "invalid JSON", None)

    if not isinstance(data, dict):
        # A message that is valid JSON but not an object was never a candidate message
        # to begin with -- same failure bucket as JSON that doesn't parse at all.
        return WsError("invalid_json", "invalid JSON", None)

    # `.get` already treats a missing key and an explicit `null` alike (both `None`) --
    # "a field whose value is JSON null counts as absent" applies to `type` too.
    type_value = data.get("type")
    if not _is_json_string(type_value):
        # request_type stays None: it is only ever "the client's type when it was a
        # readable string", which a missing/null/number/bool/array/object never was.
        return WsError("invalid_field", "type must be a string", None)
    if not is_utf8_encodable(type_value):
        # A lone surrogate from a \uXXXX escape (BC-32) is a "readable string" as far
        # as `_is_json_string` is concerned, but echoing it back as `unknown_type`'s
        # `request_type` would hand `ws_server.py` a string its own frame encode can't
        # send. Checked before dispatch, so request_type stays None here too.
        return WsError("invalid_field", "type must be valid UTF-8 text", None)

    if type_value == Cancel.TYPE:
        return Cancel()
    parser = _PARSERS.get(type_value)
    if parser is None:
        return WsError("unknown_type", "unknown type", type_value)
    return parser(data)
