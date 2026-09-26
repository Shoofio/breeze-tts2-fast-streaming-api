"""`ws_messages` tests (tasks.md T071, US4, written before `breeze_infer/ws_messages.py`
exists -- TDD red: this file is expected to fail at collection with an `ImportError`
until T075 lands).

`ws_messages.parse(raw: str) -> Message | WsError` is pure (no I/O, no session state): it
turns one client text frame into a typed dataclass (`Start`, `Text`, `Flush`, `End`,
`Instruction`, `Cancel`) or a `WsError(code, message, request_type)`. Everything that needs
session state to answer -- `not_started` (needs to know whether `start` already happened),
`unknown_voice` (needs the voice registry), `text_too_long` (needs the existing buffer
length) -- belongs to `ws_session`/`ws_server` (T072/T073/T076/T077), not here.

Ranges and grammar mirror `contracts/ws-api.md`'s "`start` details" and the field table's
"same types, ranges and defaults as HTTP", cross-checked against `http_fields.py`'s
`parse_speech` (data-model.md `SpeechRequest`): `cfg_scale` [0, 100]; `seed`
[0, 4294967295]; `split_chars` [0, 10000] (0 means no splitting, BC-38 -- unlike the
optional sampling fields, it is never a "means default" sentinel); `temperature`
(0, 10]; `top_p` (0, 1]; `repetition_penalty` [0.0001, 10]; `top_k` [1, 10000];
`max_new_tokens` [1, `MAX_NEW_TOKENS_CEILING`] (1,500). Boundary literals for the
precision-exact cases (`"100.0000000000000001"`, etc.) are copied from
`test_http_fields.py`'s `test_bc_03_decimal_range_bound_is_exact` -- the same float
double-precision rounding trap applies here, since the wire format is JSON, not a
url-encoded decimal string, but a JSON number literal has exactly the same "float() rounds
this into range" failure mode.

Numeric-precision and control-character cases build the raw JSON text by hand (not via
`json.dumps`) so the exact source literal reaches `parse` unrounded by an intermediate
Python `float`/`str` round-trip -- `json.dumps(10.0000000000000001)` would already have
lost the precision this is trying to test *before* the wire text is ever built.

Five things this file's first draft flagged as ambiguous were resolved by the coordinator
against an amended contracts/ws-api.md (see "Design notes" at the bottom for the two that
still needed an assumption pinned down, and every `# resolved:` comment inline).

See the bottom of this file ("Design notes") for the assumption the zero-sentinel and
precision-exact tests rely on.
"""

from __future__ import annotations

import pytest

from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from breeze_infer.ws_messages import (
    Cancel,
    End,
    Flush,
    Instruction,
    Message,
    Start,
    Text,
    WsError,
    parse,
)


def _start(**literals: str) -> str:
    """Raw JSON text for a `start` message. Each keyword's value must already be valid
    JSON text (`'"alice"'` for a string field, `'5'` for a number, `'true'` for a bool),
    not a Python value -- this is what lets the precision-exact boundary tests below put
    an exact source literal on the wire instead of a Python `float`'s rounded repr.
    """
    body = ",".join(f'"{name}":{literal}' for name, literal in literals.items())
    return '{"type":"start"' + ("," + body if body else "") + "}"


def _msg(type_name: str, **literals: str) -> str:
    """Raw JSON text for any other message type, same convention as `_start`."""
    body = ",".join(f'"{name}":{literal}' for name, literal in literals.items())
    return f'{{"type":"{type_name}"' + ("," + body if body else "") + "}"


def _assert_error(raw: str, code: str, *, request_type: str | None) -> WsError:
    result = parse(raw)
    assert isinstance(result, WsError)
    assert result.code == code
    assert result.request_type == request_type
    return result


# --- BC-32: a real JSON parser, not the C++ hand-rolled reader ----------------------


def test_bc_32_unicode_escapes_are_decoded() -> None:
    """BC-32: the C++ server's hand-rolled JSON reader deleted `\\uXXXX` escapes instead
    of decoding them, so an escaped non-ASCII character never survived. A real JSON
    parser decodes them correctly.
    """
    raw = '{"type":"text","text":"\\u4f60\\u597d"}'

    result = parse(raw)

    assert result == Text(text="你好")


def test_bc_32_type_like_substring_inside_a_value_is_not_the_message_type() -> None:
    """BC-32: the C++ reader matched field names anywhere in the raw message text,
    including inside string values, so a text value that happened to contain something
    that looked like `"type":"start"` could be misread as the message's own type. A real
    JSON parser parses structurally and never confuses a value's contents with a key.
    """
    raw = '{"type":"text","text":"he said \\"type\\":\\"start\\" out loud"}'

    result = parse(raw)

    assert result == Text(text='he said "type":"start" out loud')


# --- invalid_json --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{",
        "not json at all",
        '{"type": "start",}',  # trailing comma
        '{"type": "start"',  # unterminated
        "{'type': 'start'}",  # single quotes aren't valid JSON
        "undefined",
    ],
)
def test_invalid_json(raw: str) -> None:
    _assert_error(raw, "invalid_json", request_type=None)


@pytest.mark.parametrize("raw", ["[1,2,3]", "42", '"just a string"', "null", "true"])
def test_non_object_json_gets_invalid_json(raw: str) -> None:
    """Syntactically valid JSON that isn't an object was never a valid message to begin
    with -- see "Spec ambiguities" at the bottom: the contract doesn't say which code
    this gets, so this asserts the reading this suite settled on (`invalid_json`, the
    same bucket as unparseable text) rather than guessing at `invalid_field` silently.
    """
    _assert_error(raw, "invalid_json", request_type=None)


# --- missing / non-string `type` ------------------------------------------------------


def test_missing_type_gets_invalid_field() -> None:
    _assert_error('{"text": "hi"}', "invalid_field", request_type=None)


@pytest.mark.parametrize("type_literal", ["123", "true", "false", "[]", "{}", "1.5"])
def test_non_string_type_gets_invalid_field(type_literal: str) -> None:
    """`request_type` is `None` here, not the raw literal: the interface only ever
    surfaces `request_type` when the client's `type` was "a readable string", which a
    number/bool/array/object is not.
    """
    raw = '{"type":' + type_literal + "}"

    _assert_error(raw, "invalid_field", request_type=None)


# --- unknown_type ----------------------------------------------------------------------


def test_unknown_type_gets_unknown_type_with_message() -> None:
    result = parse('{"type":"bogus_message"}')

    assert isinstance(result, WsError)
    assert result.code == "unknown_type"
    assert result.message == "unknown type"
    assert result.request_type == "bogus_message"


# --- `start`: minimal message, and the additive HTTP-shared fields --------------------


def test_start_with_only_type_leaves_every_optional_field_unset() -> None:
    result = parse(_start())

    assert result == Start(
        voice_id=None,
        instruction=None,
        ref_text=None,
        cfg_scale=None,
        seed=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=None,
        split_chars=None,
    )


def test_start_accepts_top_p_repetition_penalty_and_max_new_tokens() -> None:
    """The additive fields (spec.md "Additive changes"): `top_p`, `repetition_penalty`
    and `max_new_tokens` are accepted on WebSocket `start`, same as HTTP.
    """
    raw = _start(
        voice_id='"alice"',
        top_p="0.9",
        repetition_penalty="1.1",
        max_new_tokens="500",
    )

    result = parse(raw)

    assert isinstance(result, Start)
    assert result.voice_id == "alice"
    assert result.top_p == 0.9
    assert result.repetition_penalty == 1.1
    assert result.max_new_tokens == 500


# --- numeric fields: type and range, sharing http_fields's rules ----------------------

# "Just in"/"just out" boundary literals, one field at a time. The precision-exact ones
# (a literal that rounds to the boundary as a Python `float` but isn't equal to it as a
# `Decimal`) are copied from test_http_fields.py's test_bc_03_decimal_range_bound_is_exact
# -- the shared-implementation instruction (T075) asks `ws_messages` to reuse the range
# helpers from `http_fields`, so the same exactness bug (or its absence) applies here too.

_VALID_BOUNDARIES: list[tuple[str, str, object]] = [
    ("cfg_scale", "0", 0.0),
    ("cfg_scale", "100", 100.0),
    ("seed", "0", 0),
    ("seed", "4294967295", 4294967295),
    ("split_chars", "10000", 10000),
    ("temperature", "0.0001", 0.0001),
    ("temperature", "10", 10.0),
    ("top_p", "0.0001", 0.0001),
    ("top_p", "1", 1.0),
    ("repetition_penalty", "0.0001", 0.0001),
    ("repetition_penalty", "10", 10.0),
    ("top_k", "1", 1),
    ("top_k", "10000", 10000),
    ("max_new_tokens", "1", 1),
    ("max_new_tokens", str(MAX_NEW_TOKENS_CEILING), MAX_NEW_TOKENS_CEILING),
]


@pytest.mark.parametrize(("field", "literal", "expected"), _VALID_BOUNDARIES)
def test_start_numeric_field_accepts_boundary_values(
    field: str, literal: str, expected: object
) -> None:
    result = parse(_start(voice_id='"alice"', **{field: literal}))

    assert isinstance(result, Start)
    assert getattr(result, field) == expected


_INVALID_BOUNDARIES: list[tuple[str, str]] = [
    ("cfg_scale", "-1"),
    ("cfg_scale", "101"),
    ("cfg_scale", "100.0000000000000001"),  # rounds to 100.0 as a float; exact bound is 100
    ("seed", "-1"),
    ("seed", "4294967296"),
    ("split_chars", "10001"),
    ("temperature", "-1"),
    ("temperature", "10.0000000000000001"),
    ("top_p", "-0.5"),
    ("top_p", "1.00000000000000001"),
    ("repetition_penalty", "0.00009999999999999999999"),  # just under 1e-4
    ("repetition_penalty", "10.0000000000000001"),
    ("repetition_penalty", "11"),
    ("top_k", "-5"),
    ("top_k", "10001"),
    ("max_new_tokens", str(MAX_NEW_TOKENS_CEILING + 1)),
    ("max_new_tokens", "999999"),
]


@pytest.mark.parametrize(("field", "literal"), _INVALID_BOUNDARIES)
def test_start_numeric_field_rejects_out_of_range_values(field: str, literal: str) -> None:
    _assert_error(
        _start(voice_id='"alice"', **{field: literal}), "invalid_field", request_type="start"
    )


# --- resolved: FR-006's "0 means the model default" is `parse`'s job, for the same five
# optional sampling fields `http_fields._optional_decimal`/`_optional_int` apply it to --
# `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`. `split_chars`,
# `cfg_scale` and `seed` are unaffected (BC-38 makes split_chars 0 mean "no splitting",
# not "default"; cfg_scale and seed simply accept 0 as an ordinary in-range value) -- see
# `test_bc_38_split_chars_zero_means_no_splitting_negative_is_invalid` above.
#
# ASSUMPTION this suite pins down (per the coordinator's ask to say which approach is
# assumed): `json.loads` hands a plain `parse` only a Python `int`/`float` for a JSON
# number, with no way back to the source digits -- `float("1e-400")` is indistinguishable
# from `float("0")` once parsed. `http_fields._is_zero_literal`/`_check_decimal_range`
# need the *literal text* to tell them apart (a nonzero literal that underflows to `0.0`,
# `test_underflowing_nonzero_literal_is_400_not_default` below, must NOT be treated as the
# zero sentinel). So these tests assume `ws_messages` recovers that literal text itself --
# e.g. via `json.loads(raw, parse_int=str, parse_float=str)`, which hands every JSON
# number to `parse` as its original source string instead of a rounded Python
# `int`/`float` -- and then reuses (or mirrors) `_is_zero_literal`/`_check_decimal_range`
# on that string, exactly as `http_fields` does on a form field's raw value. If T075 takes
# a different approach that can't distinguish "0" from "1e-400", the two
# `test_underflowing_nonzero_literal_is_400_not_default` cases are the ones that will need
# to change.

_ZERO_SENTINEL_CASES: list[tuple[str, str]] = [
    ("temperature", "0"),
    ("temperature", "0.0"),
    ("temperature", "-0"),
    ("temperature", "0e10"),
    ("temperature", "-0.000e5"),
    ("top_p", "0"),
    ("top_p", "0.0"),
    ("top_p", "0.000e99"),
    ("repetition_penalty", "0"),
    ("repetition_penalty", "0.0"),
    ("repetition_penalty", "-0.0"),
    ("top_k", "0"),
    ("top_k", "-0"),
    ("max_new_tokens", "0"),
    ("max_new_tokens", "-0"),
]


@pytest.mark.parametrize(("field", "literal"), _ZERO_SENTINEL_CASES)
def test_zero_literal_means_default_for_optional_sampling_fields(
    field: str, literal: str
) -> None:
    """FR-006, mirrored from `http_fields._optional_decimal`/`_optional_int`: any literal
    whose only significant digits are zero -- however signed, however written as a
    decimal or with an exponent -- means "use the model default", not "use zero".
    `_is_zero_literal` decides this from the digits, not the parsed float/int value, which
    is exactly why `-0`, `0.0` and `0e10` all count the same as a bare `0`.
    """
    result = parse(_start(voice_id='"alice"', **{field: literal}))

    assert isinstance(result, Start)
    assert getattr(result, field) is None


@pytest.mark.parametrize("field", ["temperature", "top_p", "repetition_penalty"])
def test_underflowing_nonzero_literal_is_invalid_field_not_default(field: str) -> None:
    """Review finding this suite copies from test_http_fields.py's
    `test_underflowing_nonzero_literal_is_400_not_default`: a nonzero literal tiny enough
    to underflow to exactly `0.0` as a Python `float` (`float("1e-400") == 0.0`) must
    still be judged by its own sign and digits, not mistaken for the zero sentinel --
    `1e-400` is a genuinely positive value outside `temperature`'s `(0, 10]` (`0` itself
    isn't in that range either, hence the sentinel bypass existing at all; a value that
    both underflows *and* isn't literally zero belongs to neither the sentinel case nor
    a normal in-range value, so it must fail range-checking instead of being silently
    accepted as the default).
    """
    _assert_error(
        _start(voice_id='"alice"', **{field: "1e-400"}), "invalid_field", request_type="start"
    )


# --- wrong field types (including bools, which must not be accepted as numbers) --------

_WRONG_TYPE_CASES: list[tuple[str, str]] = [
    ("voice_id", "123"),
    ("voice_id", "true"),
    ("voice_id", "[]"),
    ("instruction", "123"),
    ("instruction", "false"),
    ("ref_text", "123"),
    ("cfg_scale", '"1.0"'),
    ("cfg_scale", "true"),
    ("cfg_scale", "false"),
    ("seed", '"42"'),
    ("seed", "1.5"),  # a float literal for an integer field
    ("seed", "true"),
    ("temperature", '"0.5"'),
    ("temperature", "true"),
    ("top_k", "1.5"),
    ("top_k", "true"),
    ("top_p", '"0.5"'),
    ("top_p", "true"),
    ("repetition_penalty", '"1.0"'),
    ("repetition_penalty", "true"),
    ("max_new_tokens", '"100"'),
    ("max_new_tokens", "true"),
    ("split_chars", '"100"'),
    ("split_chars", "true"),
    ("split_chars", "1.5"),
]


@pytest.mark.parametrize(("field", "literal"), _WRONG_TYPE_CASES)
def test_start_field_wrong_type_gets_invalid_field(field: str, literal: str) -> None:
    """Covers "wrong field types" generally, and specifically that JSON `true`/`false`
    are never accepted where a number is expected -- `bool` is a subclass of `int` in
    Python, so a naive `isinstance(value, (int, float))` check would wrongly accept them
    unless the parser explicitly excludes `bool` first.
    """
    # `voice_id` defaults to a valid string so BC-13 never fires for an unrelated field
    # under test -- except when `voice_id` itself is the field under test, where setting
    # it twice would be a duplicate keyword argument, not a meaningful test.
    literals = {field: literal} if field == "voice_id" else {"voice_id": '"alice"', field: literal}

    _assert_error(_start(**literals), "invalid_field", request_type="start")


# --- resolved: an explicit JSON null on an optional field counts as absent -----------
#
# contracts/ws-api.md, "Client -> server": "a field whose value is JSON `null` counts as
# absent." SillyTavern is a real client that does this (`JSON.stringify(NaN)` is `null`,
# and it can send `cfg_scale`/`voice_id` that way), so this isn't just a hypothetical.

_EVERY_OPTIONAL_START_FIELD = [
    "voice_id",
    "instruction",
    "ref_text",
    "cfg_scale",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "split_chars",
]


@pytest.mark.parametrize("field", _EVERY_OPTIONAL_START_FIELD)
def test_start_null_field_counts_as_absent(field: str) -> None:
    result = parse(_start(**{field: "null"}))

    assert isinstance(result, Start)
    assert getattr(result, field) is None


@pytest.mark.parametrize("message_type", ["flush", "end"])
def test_flush_end_null_text_counts_as_absent(message_type: str) -> None:
    result = parse(_msg(message_type, text="null"))

    assert result == (Flush(text=None) if message_type == "flush" else End(text=None))


# --- BC-38: split_chars 0 means no splitting; negative is invalid --------------------


def test_bc_38_split_chars_zero_means_no_splitting_negative_is_invalid() -> None:
    """BC-38: the C++ server treated `split_chars <= 0` at `start` as "use 600" (its
    clamp-to-default bug, also present for 0 specifically). The new server keeps `0`
    itself to mean "no length splitting" -- it is a real, distinct value here, not a
    "use the default" sentinel the way `temperature`/`top_p`/etc.'s `0` is -- and only a
    negative value is rejected.
    """
    zero_result = parse(_start(voice_id='"alice"', split_chars="0"))
    assert isinstance(zero_result, Start)
    assert zero_result.split_chars == 0

    _assert_error(
        _start(voice_id='"alice"', split_chars="-1"), "invalid_field", request_type="start"
    )


# --- BC-46: control characters rejected ------------------------------------------------

# Same disallowed set as test_http_fields.py's _DISALLOWED_CONTROL_CHARS: NUL, BEL, ESC,
# DEL and NEL, written as \uXXXX escapes so the *wire text* stays valid JSON (Python's
# `json.loads` itself rejects a literal unescaped control byte inside a string with
# `strict=True`, its default -- so this exercises the has_control_characters check
# downstream of a successful JSON parse, not JSON syntax rejection).
_DISALLOWED_CONTROL_ESCAPES = ["\\u0000", "\\u0007", "\\u001b", "\\u007f", "\\u0085"]


@pytest.mark.parametrize("escape", _DISALLOWED_CONTROL_ESCAPES)
def test_bc_46_control_characters_rejected(escape: str) -> None:
    """BC-46: the C++ server accepted raw control characters in text -- and even gave NUL
    special meaning as sentence-closing punctuation. The new server rejects any Unicode
    `Cc` control character other than tab, CR and LF, the same rule `http_fields`/
    `text_rules.has_control_characters` already applies on HTTP.
    """
    raw = _msg("text", text=f'"hello{escape}there"')

    _assert_error(raw, "invalid_field", request_type="text")


@pytest.mark.parametrize("escape", ["\\t", "\\r", "\\n"])
def test_bc_46_tab_cr_lf_are_allowed_in_text(escape: str) -> None:
    raw = _msg("text", text=f'"hello{escape}there"')

    result = parse(raw)

    assert isinstance(result, Text)
    assert "hello" in result.text and "there" in result.text


# resolved: BC-46 also applies to `instruction` and `ref_text`, on `start` and on the
# `instruction` message -- ws-api.md's "`text` details" now says so explicitly ("as on
# HTTP"), and `http_fields._check_no_control_characters` is in fact the exact same call
# for `text`, `instruction` and `ref_text` (text_rules.has_control_characters has no
# per-field variation at all), so tab/CR/LF are allowed in these two fields for the same
# reason they're allowed in `text`.


@pytest.mark.parametrize("escape", _DISALLOWED_CONTROL_ESCAPES)
def test_bc_46_control_characters_rejected_in_start_instruction(escape: str) -> None:
    raw = _start(instruction=f'"speak{escape}slowly"')

    _assert_error(raw, "invalid_field", request_type="start")


@pytest.mark.parametrize("escape", _DISALLOWED_CONTROL_ESCAPES)
def test_bc_46_control_characters_rejected_in_start_ref_text(escape: str) -> None:
    raw = _start(voice_id='"alice"', ref_text=f'"hello{escape}world"')

    _assert_error(raw, "invalid_field", request_type="start")


@pytest.mark.parametrize("escape", _DISALLOWED_CONTROL_ESCAPES)
def test_bc_46_control_characters_rejected_in_instruction_message(escape: str) -> None:
    raw = _msg("instruction", instruction=f'"speak{escape}slowly"')

    _assert_error(raw, "invalid_field", request_type="instruction")


_JSON_ESCAPE_TO_CHAR = {"\\t": "\t", "\\r": "\r", "\\n": "\n"}


@pytest.mark.parametrize("escape", ["\\t", "\\r", "\\n"])
def test_bc_46_tab_cr_lf_are_allowed_in_start_instruction_and_ref_text(escape: str) -> None:
    char = _JSON_ESCAPE_TO_CHAR[escape]

    instruction_result = parse(_start(instruction=f'"speak{escape}slowly"'))
    assert isinstance(instruction_result, Start)
    assert instruction_result.instruction == f"speak{char}slowly"

    ref_text_result = parse(_start(voice_id='"alice"', ref_text=f'"hello{escape}world"'))
    assert isinstance(ref_text_result, Start)
    assert ref_text_result.ref_text == f"hello{char}world"


# --- BC-13: start's ref_text without voice_id is an error ------------------------------


def test_bc_13_start_ref_text_without_voice_id_is_an_error() -> None:
    """BC-13: the C++ server silently ignored a `ref_text` sent without a `voice_id`
    (there is no `ref_audio` message field over WebSocket -- only a saved voice can be
    referenced), leaving the client thinking its transcript was in effect when it never
    was. The new server rejects it instead, per contracts/ws-api.md's "start details":
    "`ref_text` comes without `voice_id`" -> `reference_required`.
    """
    raw = _start(ref_text='"a transcript that should require a voice_id"')

    result = parse(raw)

    assert isinstance(result, WsError)
    assert result.code == "reference_required"
    # The exact message isn't specified for this row (unlike unknown_voice's, which the
    # contract spells out as "unknown voice_id") -- only the code is asserted.


def test_start_with_voice_id_and_ref_text_is_accepted_by_parse() -> None:
    """The BC-13 check is co-presence only; whether `voice_id` actually names a
    registered voice needs the registry, so `unknown_voice` is out of scope for this
    pure parser (ws_session/ws_server's job, T072/T073/T076/T077)."""
    result = parse(_start(voice_id='"alice"', ref_text='"a transcript"'))

    assert isinstance(result, Start)
    assert result.voice_id == "alice"
    assert result.ref_text == "a transcript"


# --- BC-02: on start, an empty string counts as absent, same as on HTTP ----------------


def test_bc_02_start_empty_voice_id_counts_as_absent() -> None:
    """BC-02: the C++ server used an empty field value literally (`voice_id=""` looked
    up as a real, empty voice id); the new server treats it as unset, same as HTTP.
    """
    result = parse(_start(voice_id='""'))

    assert isinstance(result, Start)
    assert result.voice_id is None


def test_bc_02_start_empty_instruction_counts_as_absent() -> None:
    """BC-02: an empty `instruction` means unset here (`ws_session` then applies the
    default), the same field-level rule as HTTP's BC-02 -- distinct from BC-37, which is
    about the standalone `instruction` message and lives in `ws_session` instead.
    """
    result = parse(_start(instruction='""'))

    assert isinstance(result, Start)
    assert result.instruction is None


def test_bc_02_start_empty_ref_text_counts_as_absent_and_is_not_a_bc_13_error() -> None:
    """BC-02: an empty `ref_text` is unset, same as omitting it entirely -- so, unlike a
    real, non-blank `ref_text` sent without `voice_id`, this is not the BC-13 error.
    """
    result = parse(_start(ref_text='""'))

    assert isinstance(result, Start)
    assert result.ref_text is None
    assert result.voice_id is None


@pytest.mark.parametrize("message_type", ["text", "flush", "end"])
def test_text_flush_end_empty_string_is_kept_not_absent(message_type: str) -> None:
    """Unlike `start`'s fields, BC-02 does not extend to `text`/`flush`/`end`'s `text`:
    an empty text chunk is a real (if useless) piece of text, kept as `""`, not `None`.
    """
    result = parse(_msg(message_type, text='""'))

    expected: Message = {"text": Text(text=""), "flush": Flush(text=""), "end": End(text="")}[
        message_type
    ]
    assert result == expected


# --- the other message types: shape only ----------------------------------------------


def test_text_message_parses() -> None:
    assert parse(_msg("text", text='"hello there"')) == Text(text="hello there")


def test_flush_without_text() -> None:
    assert parse(_msg("flush")) == Flush(text=None)


def test_flush_with_text() -> None:
    assert parse(_msg("flush", text='"more text"')) == Flush(text="more text")


def test_end_without_text() -> None:
    assert parse(_msg("end")) == End(text=None)


def test_end_with_text() -> None:
    assert parse(_msg("end", text='"final words"')) == End(text="final words")


def test_instruction_message_parses() -> None:
    assert parse(_msg("instruction", instruction='"Speak angrily."')) == Instruction(
        instruction="Speak angrily."
    )


def test_instruction_message_allows_a_blank_instruction() -> None:
    """Resetting to the default on a blank instruction is BC-37, a session-level rule
    (tasks.md T072, `test_bc_37_blank_instruction_resets_to_default`) -- `parse` just
    passes the string through unchanged, blank or not.
    """
    assert parse(_msg("instruction", instruction='""')) == Instruction(instruction="")


def test_cancel_message_has_no_fields() -> None:
    assert parse(_msg("cancel")) == Cancel()


def test_unknown_fields_are_ignored() -> None:
    """contracts/ws-api.md: "Unknown fields are ignored" -- a field this contract never
    defined must never itself cause invalid_field."""
    raw = '{"type":"text","text":"hi","this_field_does_not_exist":123}'

    assert parse(raw) == Text(text="hi")


# --- Design notes -----------------------------------------------------------------------
#
# This file's first draft flagged five things as ambiguous; the coordinator resolved all
# five against an amended contracts/ws-api.md. Kept as confirmed choices (unchanged from
# the first draft):
#
# 1. Non-object top-level JSON (`[1,2,3]`, `42`, `"a string"`, `null`, `true`) ->
#    `invalid_json` (test_non_object_json_gets_invalid_json). ws-api.md now says so
#    explicitly: "A message that is valid JSON but not an object gets `invalid_json`".
# 2. A JSON object with no `type` key, or a non-string `type` -> `invalid_field`
#    (test_missing_type_gets_invalid_field, test_non_string_type_gets_invalid_field).
#    ws-api.md now says so explicitly too: "a missing or non-string `type` gets
#    `invalid_field`".
#
# Resolved with new behavior added (tests above, not just confirmed):
#
# 3. An explicit JSON `null` on any optional field counts as absent (`None`), not a
#    type error -- ws-api.md: "a field whose value is JSON `null` counts as absent".
#    See test_start_null_field_counts_as_absent and test_flush_end_null_text_counts_as_
#    absent. (Motivation, not just a hypothetical: SillyTavern can send `cfg_scale: null`
#    from `JSON.stringify(NaN)`.)
# 4. BC-46 applies to `instruction` and `ref_text` too, on `start` and on the standalone
#    `instruction` message, same as HTTP -- ws-api.md's "`text` details" now says "as on
#    HTTP" explicitly. See test_bc_46_control_characters_rejected_in_start_instruction,
#    _in_start_ref_text, _in_instruction_message, and the matching tab/CR/LF-allowed test.
# 5. FR-006's "0 means the model default" (-> `None`) is `parse`'s own job, for the same
#    five fields `http_fields._optional_decimal`/`_optional_int` apply it to. See the
#    "resolved: FR-006" comment block and its tests above (the exactness ASSUMPTION this
#    suite pins down -- that `parse` recovers each number's literal source text rather
#    than working from the parsed Python `float`/`int` alone -- is spelled out there in
#    full, since it's the one place this file commits to *how* T075 must implement this,
#    not just *what* the observable behavior should be).
