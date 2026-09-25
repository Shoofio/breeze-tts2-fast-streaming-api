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
duplicate field, and its parser limits can't be changed. Routes call `request.form()`
themselves instead. research.md R6: a truncated multipart body parses as an empty form
with status 200, so `text` (and every other required field) has to be checked explicitly
here rather than relying on the parser to reject a short body.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from starlette.datastructures import FormData, QueryParams, UploadFile

from breeze_infer.errors import ApiError
from breeze_infer.settings import Settings

# contracts/http-api.md "Fields": the defaults for POST /v1/audio/speech.
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
DEFAULT_CFG_SCALE = 1.0
DEFAULT_SEED = 42

# research.md R6: the limits every route passes to `request.form()`.
FORM_MAX_FILES = 1
FORM_MAX_FIELDS = 32
FORM_MAX_PART_SIZE = 64 * 1024


@dataclass(frozen=True)
class Fields:
    """The request's fields, still as strings, kept source-separated.

    `form` and `query` are both multi-dicts (`getlist` gives every value for a name), kept
    apart rather than merged into one mapping so T048's duplicate check -- "`getlist(k)` has
    more than one value in the form or the query string, or the same key appears in both"
    (research.md R6) -- can be built directly from them instead of reconstructing which
    source each value came from.

    `ref_audio`'s file part is read into bytes here, the only place that awaits the part's
    body -- an `UploadFile`'s backing `SpooledTemporaryFile` is only guaranteed to survive
    for the request's lifetime, and by the time `parse_speech` runs synchronously that
    lifetime guarantee is the wrong shape to rely on. Decoding those bytes into audio (R9,
    `reference_audio.decode`) stays the route's job, run after the field and reference
    checks (FR-007's order).
    """

    form: FormData
    query: QueryParams
    ref_audio: bytes | None

    def values(self, name: str) -> list[str]:
        """Every string value given for `name`, from the form and the query combined.

        Only ever called for the non-file fields (`ref_audio` is read separately into
        `self.ref_audio`), so every value here really is a string, not an `UploadFile` --
        the `str(...)` is just to satisfy the type checker, which only knows `FormData`
        values as `UploadFile | str` in general.
        """
        return [str(v) for v in self.form.getlist(name)] + [*self.query.getlist(name)]


async def read_fields(request: Request) -> Fields:
    """Parse the body (bounded per research.md R6) and merge it with the query string."""
    form = await request.form(
        max_files=FORM_MAX_FILES,
        max_fields=FORM_MAX_FIELDS,
        max_part_size=FORM_MAX_PART_SIZE,
    )
    ref_audio: bytes | None = None
    for value in form.getlist("ref_audio"):
        if isinstance(value, UploadFile):
            ref_audio = await value.read()

    return Fields(form=form, query=request.query_params, ref_audio=ref_audio)


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
    fields: Fields, name: str, parser: Callable[[str, str], float | int]
) -> float | int | None:
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
