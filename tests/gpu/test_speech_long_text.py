"""A long voice-design request through the real HTTP route and the real runtime (tasks.md T051,
porting `api-alignment:tests/gpu/test_speech_long_text.py`; SC-004: "A 3000-character request with
no reference completes with audio covering the whole text in 100% of 5 runs").

Built the same way `tests/gpu/test_speech_http.py` builds its app -- `create_app` over the
session-scoped `gpu_env` fixture from `tests/gpu/conftest.py`, with a fresh `Components` (gate,
GPU thread, `RecordingEvents`) per run so no lease or event can leak between runs -- rather than
the old `A:` file's module-level `breeze_infer.api.app`, which this branch no longer has
(`breeze_infer/api.py` is a composition root only, per tasks.md T022).

Anchoring (tasks.md T050/T052/T053, commit `6bb3574`) has landed: with no reference and more than
one piece, `breeze_infer/routes_speech.py`'s `_iter_pieces` collects piece 0's generated frames
and turns them into a `synthesis.CodesRef` (`anchor_codes`) that every later piece is prepared
against, so one speaker is heard throughout instead of a fresh voice design per piece, unless the
decision instead declines it (`speech.anchor_skipped`, checked for directly below). There is no
event for a *successful* anchor, though -- `breeze_infer/events.py` has `speech.piece_done
{piece_index, frames}`, `speech.piece_clamped` and `speech.anchor_skipped` (the decline), but
nothing like `speech.anchored` for the normal case -- so this test spies on
`routes_speech.prepare_piece`'s `reference` argument (the same technique `tests/test_long_text.py`
uses against `FakeRuntime`) to observe it directly: piece 0 must be prepared with `NoRef` (voice
design) and every later piece with the `CodesRef` anchoring produced. If a future reviewer wants a
GPU test that doesn't need to reach into route internals, `routes_speech.py` would need to emit
something like `speech.anchored{piece, frames}` next to `anchor_codes`'s call site -- flagged here
rather than added, since that's an observability change to production code, not this test.
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from breeze_infer import routes_speech
from breeze_infer.api import create_app
from breeze_infer.limits import ANCHOR_CHARS
from breeze_infer.settings import DEFAULT_SPLIT_CHARS
from breeze_infer.streaming import BYTES_PER_SAMPLE
from breeze_infer.synthesis import CodesRef, NoRef, Reference, anchor_sizing
from breeze_infer.synthesis import prepare_piece as _real_prepare_piece
from breeze_infer.text_split import split_text
from models.fast_streaming import prompt_length
from tests.gpu.test_speech_http import (
    SAMPLE_RATE,
    SPEECH_PATH,
    _components,
    _wait_for_terminal_event,
    shut_down_cpu_tokenizer,
)

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV_PATH = (
    REPO_ROOT / "specs" / "003-cpp-compatible-api" / "research" / "long-text-sample.wav"
)

RUNS = 5

# A fixed ~3,000-character English passage, no reference audio: voice design only, split with
# the server's default `split_chars` (600, so this lands as several pieces).
LONG_PASSAGE = (
    "The lighthouse keeper climbed the spiral staircase before dawn, counting each worn "
    "stone step out of habit rather than necessity. Fog had settled over the harbor "
    "during the night, thick enough to blur the outline of the fishing boats moored "
    "along the quay. He had made this climb every morning for eleven years, and in "
    "that time he had learned to read the weather from the smell of the salt air alone. "
    "This morning it smelled like rain, though the sky above the water still held a "
    "handful of pale stars.\n\n"
    "At the top of the tower, he lit the lamp and checked the mechanism that turned "
    "the great lens, listening for the faint click that told him the gears were "
    "properly seated. A ship had run aground on the rocks two miles north the previous "
    "winter, and the keeper had never quite forgiven himself for the ten minutes the "
    "light had gone dark while he repaired a frayed wire. Since then he inspected "
    "every joint and bolt twice, once by sight and once by touch, before he allowed "
    "himself to sit down with his coffee and watch the horizon lighten.\n\n"
    "The village below was still asleep, its narrow streets empty except for a baker "
    "who was already opening his shutters and a dog trotting along the seawall with "
    "no particular destination in mind. Smoke began to rise from a handful of chimneys "
    "as families woke and lit their stoves against the morning chill. The keeper could "
    "see all of this from his perch, tiny and distant, like a model village built for "
    "a child rather than a town where real people lived and worked and worried about "
    "the price of fish.\n\n"
    "By midmorning the fog had burned away, and the fishing boats began to leave the "
    "harbor one by one, their engines coughing to life in a ragged chorus. The keeper "
    "waved from the gallery at the top of the tower, an old ritual that none of the "
    "younger fishermen understood but that the older ones still returned out of "
    "respect. He watched until the last boat cleared the headland, then climbed back "
    "down to record the morning's weather in a logbook that stretched back further "
    "than anyone in the village could remember.\n\n"
    "In the afternoon he walked into the village to buy bread and to hear whatever "
    "news had traveled up the coast since the previous day. The baker's wife told him "
    "about a wedding planned for the following month, and an old fisherman argued that "
    "the mackerel were running earlier than usual this year, a sign, he insisted, of a "
    "hard winter to come. The keeper listened with the patient attention of a man who "
    "had heard a thousand such predictions and had learned that the sea rarely cared "
    "what anyone expected of it.\n\n"
    "As evening approached he climbed the tower once more, his knees aching a little "
    "more than they had that morning, and lit the lamp again as the light began to "
    "fail. The lens turned, throwing its beam out across the darkening water in slow, "
    "patient sweeps, the same rhythm it had kept every night since long before he was "
    "born. He stayed at the top of the tower until the last color drained from the "
    "sky, watching the beam do its quiet work, and then he went down to sleep so that "
    "he could climb the stairs again the next morning and do it all over."
)
assert 2900 <= len(LONG_PASSAGE) <= 3200, f"passage is {len(LONG_PASSAGE)} characters"

# The server's actual splitting rule (text_split.split_text), with no reference and the
# server's default split_chars: the same opening-budget/flat-budget split
# breeze_infer.routes_speech._serve_speech itself computes. Fixed once here (LONG_PASSAGE and
# the split inputs are both constants), so every run must produce exactly this many pieces --
# a silent change in piece count (e.g. a piece merging or an extra empty piece) would drop this
# assertion rather than only be caught by a looser "at least one piece" check.
EXPECTED_PIECES = split_text(
    LONG_PASSAGE, budget=DEFAULT_SPLIT_CHARS, first_budget=min(ANCHOR_CHARS, DEFAULT_SPLIT_CHARS)
)
assert len(EXPECTED_PIECES) > 1, (
    "the passage must split into more than one piece to exercise anchoring at all"
)

# SC-004's floor: "completes with audio covering the whole text." A believable spoken rate for
# English prose is roughly 12-15 characters/second (~150 words/minute at ~5 characters/word); 25
# characters/second is far faster than any real speaker, so text_chars / 25 is a deliberately
# generous (low) floor -- it only catches a stream that's badly truncated (e.g. stops after the
# first piece or two), not one that's merely on the slower, more natural end of the range.
CHARS_PER_SECOND_FLOOR = 25.0
MIN_DURATION_SECONDS = len(LONG_PASSAGE) / CHARS_PER_SECOND_FLOOR


def _run_once(
    gpu_env, prepare_piece_spy: Any
) -> tuple[bytes, list[tuple[str, dict[str, object]]]]:
    """POST the long passage once through a fresh app/gate/thread, wait for the stream's own
    terminal event (`test_speech_http._wait_for_terminal_event`: the response body can reach the
    TestClient a few loop iterations before `SpeechResponse.__call__`'s `finally` emits the
    outcome event), and return the raw PCM body plus every event recorded.

    `prepare_piece_spy` replaces `routes_speech.prepare_piece` for the run's duration (both of
    its call sites, `_prepare_first_piece` and `_iter_pieces`, look it up by that module-level
    name), so the caller can inspect which `Reference` each piece was actually prepared with.
    """
    components, events = _components(gpu_env)
    original_prepare_piece = routes_speech.prepare_piece
    routes_speech.prepare_piece = prepare_piece_spy
    try:
        client = TestClient(create_app(components))
        response = client.post(SPEECH_PATH, data={"text": LONG_PASSAGE})
        assert response.status_code == 200, response.text
        _wait_for_terminal_event(events)
        return response.content, events.calls
    finally:
        routes_speech.prepare_piece = original_prepare_piece
        try:
            components.gpu.shutdown()
        finally:
            # Nested, not sequential (review 32, review-of-2d9070a #4/#7 in the same pass): a
            # `gpu.shutdown()` failure must not leave the CPU tokenizer's own workers running
            # past the test, which would leak their threads into the next one. It waits, with
            # a bound (review 33 on 10f0c29, review 34 finding 6): the next run's workers use
            # the same tokenizer copies (`test_speech_http._shared_loaded_model`).
            shut_down_cpu_tokenizer(components)


def _save_wav(body: bytes) -> None:
    """24 kHz mono s16 WAV, for the manual listening check in tasks.md T054. `*.wav` is
    gitignored (see `.gitignore`), so this is a local artifact only."""
    SAMPLE_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(SAMPLE_WAV_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(body)


def test_long_voice_design_completes_every_run(gpu_env) -> None:
    body_hashes: list[str] = []
    per_run_piece_frames: list[list[int]] = []

    for run_index in range(RUNS):
        prepare_calls: list[Reference] = []

        def prepare_piece_spy(
            tokenizer: Any,
            model: Any,
            reference: Reference,
            text: str,
            instruction: str,
            cfg_scale: float,
            *,
            _prepare_calls: list[Reference] = prepare_calls,
        ) -> dict[str, Any]:
            _prepare_calls.append(reference)
            return _real_prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)

        body, calls = _run_once(gpu_env, prepare_piece_spy)

        by_name: dict[str, list[dict[str, object]]] = {}
        for name, fields in calls:
            by_name.setdefault(name, []).append(fields)

        assert "speech.failed" not in by_name, f"run {run_index}: {by_name.get('speech.failed')}"
        assert "speech.aborted" not in by_name, f"run {run_index}: {by_name.get('speech.aborted')}"
        assert len(by_name.get("speech.completed", [])) == 1, (
            f"run {run_index}: expected exactly one speech.completed, got {calls}"
        )

        # Piece count: the real split (EXPECTED_PIECES, computed once at module load from the
        # same `split_text` call `_serve_speech` makes) rather than "at least one piece" -- a
        # merged or dropped piece would otherwise pass silently as long as some audio came out.
        piece_events = by_name.get("speech.piece_done", [])
        assert len(piece_events) == len(EXPECTED_PIECES), (
            f"run {run_index}: expected {len(EXPECTED_PIECES)} pieces "
            f"(split_text(budget={DEFAULT_SPLIT_CHARS}, first_budget={ANCHOR_CHARS})), "
            f"got {len(piece_events)}: {piece_events}"
        )
        assert all(fields["frames"] > 0 for fields in piece_events), (
            f"run {run_index}: a piece produced 0 frames: {piece_events}"
        )
        per_run_piece_frames.append(
            [fields["frames"] for fields in sorted(piece_events, key=lambda f: f["piece_index"])]
        )

        # Anchoring (review 32, review-of-2d9070a #1 in the same pass): confirm the anchor
        # actually held before checking what it was, so a run that quietly declined it
        # (`speech.anchor_skipped`) fails here with its reason, not on the `NoRef`-vs-`CodesRef`
        # mismatch below, which would otherwise look like the same failure for a different cause.
        skipped = by_name.get("speech.anchor_skipped", [])
        assert not skipped, f"run {run_index}: anchor was skipped: {skipped}"

        # Anchoring (T050/T052/T053): piece 0 is voice design (`NoRef`); every later piece must
        # have been prepared against the `CodesRef` `anchor_codes` built from piece 0's own
        # frames, not a fresh `NoRef` voice design each time -- see the module docstring for why
        # this spies on `prepare_piece` rather than reading an event.
        assert len(prepare_calls) == len(EXPECTED_PIECES), (
            f"run {run_index}: expected one prepare_piece call per piece "
            f"({len(EXPECTED_PIECES)}), got {len(prepare_calls)}"
        )
        assert isinstance(prepare_calls[0], NoRef), (
            f"run {run_index}: piece 0 should start as voice design (NoRef), "
            f"got {type(prepare_calls[0])}"
        )
        for piece_index, reference in enumerate(prepare_calls[1:], start=1):
            assert isinstance(reference, CodesRef), (
                f"run {run_index}: piece {piece_index} should be anchored to piece 0's audio "
                f"(CodesRef), got {type(reference)}"
            )
            assert reference.codes.shape[0] > 0, (
                f"run {run_index}: piece {piece_index}'s anchor has no frames"
            )

        assert len(body) > 0
        assert len(body) % BYTES_PER_SAMPLE == 0
        sample_count = len(body) // BYTES_PER_SAMPLE
        duration_seconds = sample_count / SAMPLE_RATE
        assert duration_seconds > MIN_DURATION_SECONDS, (
            f"run {run_index}: {duration_seconds:.2f}s of audio for {len(LONG_PASSAGE)} "
            f"characters, below the {MIN_DURATION_SECONDS:.2f}s floor "
            f"({CHARS_PER_SECOND_FLOOR} chars/s)"
        )
        samples = np.frombuffer(body, dtype="<i2")
        assert np.abs(samples).max() > 0, f"run {run_index}: audio is entirely silence"

        body_hashes.append(hashlib.sha256(body).hexdigest())

        # Each run costs real GPU minutes; print the numbers `-s` surfaces so a passing run still
        # leaves a record of what it actually generated (tasks.md T051 asks for these in the
        # report), without adding a second, silent way for a run to look fine while
        # under-producing relative to its neighbors.
        print(
            f"run {run_index}: {duration_seconds:.2f}s audio, {len(piece_events)} pieces, "
            f"floor {MIN_DURATION_SECONDS:.2f}s, sha256 {body_hashes[-1][:12]}"
        )

        if run_index == 0:
            _save_wav(body)

    # The fixed text and fixed seed (the request's default) make the *shape* of every run
    # close, but not required to be identical: the sample *values* are not bit-for-bit
    # identical run to run, even with a matched seed on an already-warmed runtime (expected of
    # CUDA kernels without `torch.use_deterministic_algorithms(True)` -- cuDNN's heuristic
    # algorithm selection, TF32 accumulation and CUDA-graph replay are not guaranteed
    # bit-reproducible across invocations -- not a bug this test should chase; forcing full
    # determinism is a runtime-wide, production change, well outside a single test file). One
    # flipped token can move a piece's EOS by a frame or more.
    #
    # Piece 0 is compared frame by frame: it is voice design from the same seed, so kernel
    # noise moves it by a frame or two at most. Every later piece is anchored to piece 0's own
    # audio (T050/T052/T053), so any change in piece 0 changes every later piece's prompt and
    # can move its length by far more than that (review 33 on 10f0c29). Each later piece is
    # compared with run 0's on its own, within 25% (review 34 finding 2), which catches one
    # piece cut short or overrun by more than a quarter. A total over all pieces could not:
    # each later piece is only 15-18% of it, and errors in opposite directions cancel out.
    # The piece count needs no check
    # here: every run already asserted exactly `len(EXPECTED_PIECES)` `speech.piece_done`
    # events (and `prepare_piece` calls) above.
    FIRST_PIECE_FRAME_TOLERANCE = 2
    LATER_PIECE_RELATIVE_TOLERANCE = 0.25
    baseline = per_run_piece_frames[0]
    all_runs = f"all runs' per-piece frames: {per_run_piece_frames}"
    for run_index, frames in enumerate(per_run_piece_frames):
        assert abs(frames[0] - baseline[0]) <= FIRST_PIECE_FRAME_TOLERANCE, (
            f"run {run_index} piece 0 has {frames[0]} frames, run 0 has {baseline[0]} "
            f"(tolerance {FIRST_PIECE_FRAME_TOLERANCE})\n{all_runs}"
        )
        mismatches = [
            (piece_index, frames[piece_index], baseline[piece_index])
            for piece_index in range(1, len(baseline))
            if abs(frames[piece_index] - baseline[piece_index])
            > LATER_PIECE_RELATIVE_TOLERANCE * baseline[piece_index]
        ]
        assert not mismatches, (
            f"run {run_index} later pieces differ from run 0 by more than "
            f"{LATER_PIECE_RELATIVE_TOLERANCE:.0%} (piece_index, run_frames, run_0_frames): "
            f"{mismatches}\n{all_runs}"
        )


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_anchor_sizing_matches_real_prompts_with_the_real_tokenizer(
    gpu_env, cfg_scale: float
) -> None:
    """The anchor check adds the anchor's length to each later piece's prompt length, measured
    without it. That rests on the real tokenizer splitting the prompt at the audio markers, so
    the sum must equal the real anchored prompt's length for every piece of the passage."""
    pieces = split_text(
        LONG_PASSAGE,
        budget=DEFAULT_SPLIT_CHARS,
        first_budget=min(ANCHOR_CHARS, DEFAULT_SPLIT_CHARS),
    )
    instruction = "A calm adult male voice, clear and natural."
    codebooks = int(gpu_env.model.config.num_codebooks)
    sizing = anchor_sizing(
        gpu_env.runtime, gpu_env.tokenizer, pieces[0], pieces[1:], instruction, cfg_scale
    )

    for frames in (1, 57, 300):
        anchor = CodesRef(
            codes=np.zeros((frames, codebooks), dtype=np.int16), ref_text=pieces[0]
        )
        for text, length in zip(pieces[1:], sizing.later_lengths, strict=True):
            real = prompt_length(
                _real_prepare_piece(
                    gpu_env.tokenizer, gpu_env.model, anchor, text, instruction, cfg_scale
                )
            )
            assert sizing.anchored(length, frames) == real
