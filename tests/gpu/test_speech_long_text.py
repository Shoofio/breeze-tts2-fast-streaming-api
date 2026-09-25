"""A long voice-design request through the real HTTP route and the real runtime (tasks.md T051,
porting `api-alignment:tests/gpu/test_speech_long_text.py`; SC-004: "A 3000-character request with
no reference completes with audio covering the whole text in 100% of 5 runs").

Built the same way `tests/gpu/test_speech_http.py` builds its app -- `create_app` over the
session-scoped `gpu_env` fixture from `tests/gpu/conftest.py`, with a fresh `Components` (gate,
GPU thread, `RecordingEvents`) per run so no lease or event can leak between runs -- rather than
the old `A:` file's module-level `breeze_infer.api.app`, which this branch no longer has
(`breeze_infer/api.py` is a composition root only, per tasks.md T022).

At the time this was written, `breeze_infer/routes_speech.py` had no anchoring yet (no
`ANCHOR_CHARS` import; tasks.md T052/T053 land it later in the same phase) and `git log --oneline
-3` topped at `196f7af` "Decide every shutdown outcome in one table (T022 final review)". Text is
still split with a flat `split_chars` budget and no first-piece anchor, so a later agent landing
T052/T053 changes how pieces are seeded/anchored but not the per-run assertions this test makes
(every stream completes with plausible, non-empty audio) -- the duration floor and piece count are
loose enough to hold either way.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from breeze_infer.api import create_app
from breeze_infer.streaming import BYTES_PER_SAMPLE
from tests.gpu.test_speech_http import (
    SAMPLE_RATE,
    SPEECH_PATH,
    _components,
    _wait_for_terminal_event,
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

# SC-004's floor: "completes with audio covering the whole text." A believable spoken rate for
# English prose is roughly 12-15 characters/second (~150 words/minute at ~5 characters/word); 25
# characters/second is far faster than any real speaker, so text_chars / 25 is a deliberately
# generous (low) floor -- it only catches a stream that's badly truncated (e.g. stops after the
# first piece or two), not one that's merely on the slower, more natural end of the range.
CHARS_PER_SECOND_FLOOR = 25.0
MIN_DURATION_SECONDS = len(LONG_PASSAGE) / CHARS_PER_SECOND_FLOOR


def _run_once(gpu_env) -> tuple[bytes, list[tuple[str, dict[str, object]]]]:
    """POST the long passage once through a fresh app/gate/thread, wait for the stream's own
    terminal event (`test_speech_http._wait_for_terminal_event`: the response body can reach the
    TestClient a few loop iterations before `SpeechResponse.__call__`'s `finally` emits the
    outcome event), and return the raw PCM body plus every event recorded."""
    components, events = _components(gpu_env)
    client = TestClient(create_app(components))
    try:
        response = client.post(SPEECH_PATH, data={"text": LONG_PASSAGE})
        assert response.status_code == 200, response.text
        _wait_for_terminal_event(events)
        return response.content, events.calls
    finally:
        components.gpu.shutdown()


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
    for run_index in range(RUNS):
        body, calls = _run_once(gpu_env)

        by_name: dict[str, list[dict[str, object]]] = {}
        for name, fields in calls:
            by_name.setdefault(name, []).append(fields)

        assert "speech.failed" not in by_name, f"run {run_index}: {by_name.get('speech.failed')}"
        assert "speech.aborted" not in by_name, f"run {run_index}: {by_name.get('speech.aborted')}"
        assert len(by_name.get("speech.completed", [])) == 1, (
            f"run {run_index}: expected exactly one speech.completed, got {calls}"
        )

        piece_events = by_name.get("speech.piece_done", [])
        assert piece_events, f"run {run_index}: no speech.piece_done events, got {calls}"
        assert all(fields["frames"] > 0 for fields in piece_events), (
            f"run {run_index}: a piece produced 0 frames: {piece_events}"
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

        # Each run costs real GPU minutes; print the numbers `-s` surfaces so a passing run still
        # leaves a record of what it actually generated (tasks.md T051 asks for these in the
        # report), without adding a second, silent way for a run to look fine while
        # under-producing relative to its neighbors.
        print(
            f"run {run_index}: {duration_seconds:.2f}s audio, {len(piece_events)} pieces, "
            f"floor {MIN_DURATION_SECONDS:.2f}s"
        )

        if run_index == 0:
            _save_wav(body)
