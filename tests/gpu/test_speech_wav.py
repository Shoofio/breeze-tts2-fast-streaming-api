"""`GET /v1/audio/speech.wav` against the real runtime (specs/004-browser-wav-stream, T006).

Built the way `test_speech_long_text.py` builds its app: `create_app` over the session's warmed
`gpu_env`, with a fresh `Components` per test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from breeze_infer.api import create_app
from breeze_infer.routes_speech import wav_header
from breeze_infer.streaming import BYTES_PER_SAMPLE
from breeze_infer.synthesis import codec_samples_per_frame
from tests.gpu.test_speech_http import (
    SPEECH_PATH,
    _components,
    _wait_for_terminal_event,
    shut_down_cpu_tokenizer,
)

pytestmark = pytest.mark.gpu

WAV_PATH = "/v1/audio/speech.wav"
FIELDS = {"text": "The lighthouse keeper climbed the stairs before dawn.", "seed": "7"}

# Same tolerance as FIRST_PIECE_FRAME_TOLERANCE in test_speech_long_text.py: GPU output is not
# bit-reproducible even with a fixed seed, so one single-piece text can differ by a frame or
# two between the GET and the POST.
FRAME_TOLERANCE = 2


def test_get_wav_matches_post_pcm_within_frame_tolerance(gpu_env) -> None:
    components, events = _components(gpu_env)
    try:
        client = TestClient(create_app(components))

        get_response = client.get(WAV_PATH, params=FIELDS)
        # Checked first: a GET refused before the GPU emits no terminal event, and the wait
        # below would hide its real status and error.
        assert get_response.status_code == 200, get_response.text
        # The GET's outcome event is emitted after its body reaches the client; the POST would
        # get a 409 if the gate were still held.
        _wait_for_terminal_event(events)
        post_response = client.post(SPEECH_PATH, data=FIELDS)
        assert post_response.status_code == 200, post_response.text
    finally:
        try:
            components.gpu.shutdown()
        finally:
            shut_down_cpu_tokenizer(components)

    assert get_response.headers["content-type"] == "audio/wav"
    sample_rate = int(get_response.headers["x-sample-rate"])
    header = wav_header(sample_rate)
    assert get_response.content[: len(header)] == header

    get_pcm = get_response.content[len(header) :]
    post_pcm = post_response.content
    for name, pcm in (("GET", get_pcm), ("POST", post_pcm)):
        assert len(pcm) > 0, f"{name} body is empty"
        assert len(pcm) % BYTES_PER_SAMPLE == 0, f"{name} body has a partial sample"

    frame_bytes = codec_samples_per_frame(gpu_env.runtime) * BYTES_PER_SAMPLE
    get_frames = len(get_pcm) / frame_bytes
    post_frames = len(post_pcm) / frame_bytes
    assert abs(get_frames - post_frames) <= FRAME_TOLERANCE, (
        f"GET has {get_frames:.2f} frames, POST has {post_frames:.2f} "
        f"(tolerance {FRAME_TOLERANCE})"
    )
