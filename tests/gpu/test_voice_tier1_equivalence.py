"""Tier 1 (stored codec tokens) must be bit-identical to the inline path, ported from
`api-alignment:tests/gpu/test_voice_tier1_equivalence.py` (tasks.md T067).

The old test built `ref_audio_path` and `ref_audio_codes` requests directly and compared their
`prepare_inputs` tensors and generated frames; `_register_voice`, `VoiceIndex` and
`breeze_infer.voices` are all gone on this branch (replaced by `routes_voices`/`voice_store`/
`voice_registry`). The port drives the comparison entirely through the real HTTP routes
instead: an inline `POST /v1/audio/speech` (`ref_audio` + `ref_text`) and a registered voice's
codes path (`voice_id` + a `ref_text` override, `synthesis.voice_reference`) both resolve to a
`CodesRef` built from the same encode of the same recording
(`breeze_infer.audio.encode_prompt_waveform`, deterministic per its own docstring), fed through
the same template with the same seed.

Both requests upload the *exact same bytes* (`ref_wav.read_bytes()`), not an ndarray this test
would re-encode to WAV a second time: a first run that read the file through `soundfile` and
re-wrote it for the registration upload diverged from the inline path by 1 sample LSB -- a
float32 round trip through 16-bit PCM shifted one reference sample by 1/32768, which changed
the *encoded reference codes* slightly (`encode_prompt_waveform` is only deterministic for
identical input samples, not merely "the same clip"). Not a code bug: `register_voice` in
test_voice_equivalence.py normally takes an ndarray it converts itself for every other GPU
voice test, where there is no independent copy of the pre-conversion bytes to drift from; this
test alone needs bit-identical bytes on both requests, so it uses `register_voice_bytes`
directly.

**What "bit-identical" actually means here, and why the PCM check below is only a sanity
check.** Even with byte-identical reference bytes, an investigation (four GPU diagnostics,
reported to and resolved with the user) found the two requests' raw PCM responses still
diverged -- and, decisively, that *two copies of the same inline request*, run back to back
through the real server, diverge the same way. Capturing the actual generated codec token
frames (by wrapping the shared runtime's `iter_audio_chunks`, the same technique
`tests/gpu/test_speech_long_text.py` uses on `prepare_piece`) showed the inline and codes-path
requests' frames are `torch.equal` in every case; only the decoded PCM differs, and by more
than rounding noise: decoding that one identical set of frames twice more, by hand, through the
streaming codec (`runtime._decode_codec_frames`) under two different request ids, 72,960 of
76,800 samples differed, max abs difference about 0.072 of the codec's [-1, 1] float output --
about 2,365 to 2,654 int16 steps (about 8% of full scale) measured across a few such pairs.
This is codec **decode**, not encode: research.md R18 already documents (and this task's own
investigation confirms with the numbers above, recorded there as an open item) that only the
reference *encode* is pinned deterministic (`benchmark=False, deterministic=True`); decode
keeps `--fast-all`'s autotuned, non-deterministic `cudnn.benchmark=True` algorithms
deliberately, since pinning every request's own decode (not just a one-time reference encode)
was judged not worth its cost -- untested here, so this is left as an open question rather
than a settled trade-off. So "bit-identical" is asserted at the level this branch's own code
actually guarantees it -- the generated token frames, checked with `torch.equal` -- and the PCM
is only sanity-checked (equal length, non-empty, not silent): a real divergence (wrong voice,
garbled audio, a different length) would still fail loudly, without this test asserting a
byte- or step-level PCM equality the runtime doesn't actually provide.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch

from tests.gpu.test_speech_http import VOICE_DIR
from tests.gpu.test_voice_equivalence import (  # noqa: F401
    register_voice_bytes,
    speak,
    voices_app,
)

pytestmark = pytest.mark.gpu

TEXT = "We need to discuss what happened last night before anyone else does."
INSTRUCTION = "Speak slowly with a restrained, serious tone."
MAX_NEW_TOKENS = 40


def _spy_iter_audio_chunks(runtime: Any) -> tuple[list[list[torch.Tensor]], Any]:
    """Wrap `runtime.iter_audio_chunks` to record each call's generated frames, in call order,
    while still running the real generator (and any token_observer the route itself passed).
    Returns `(calls, restore)`; the caller must call `restore()` once done -- `runtime` here is
    `gpu_env`'s session-scoped runtime, shared by every GPU test, so leaving it patched would
    affect tests that run after this one."""
    real_iter_audio_chunks = runtime.iter_audio_chunks
    calls: list[list[torch.Tensor]] = []

    def spy(inputs, *, token_observer=None, **kwargs):
        frames: list[torch.Tensor] = []
        calls.append(frames)

        def combined(frame: torch.Tensor) -> None:
            frames.append(frame.detach().cpu())
            if token_observer is not None:
                token_observer(frame)

        return real_iter_audio_chunks(inputs, token_observer=combined, **kwargs)

    runtime.iter_audio_chunks = spy

    def restore() -> None:
        runtime.iter_audio_chunks = real_iter_audio_chunks

    return calls, restore


def test_codes_path_matches_inline_path_exactly(voices_app) -> None:  # noqa: F811
    client, events, components = voices_app
    if VOICE_DIR is None:
        pytest.skip(
            "REFERENCE_VOICES_DIR is not set "
            "(set it to a directory containing eric/eric.wav and eric.txt)"
        )
    ref_wav = VOICE_DIR / "eric.wav"
    ref_txt = VOICE_DIR / "eric.txt"
    if not ref_wav.is_file() or not ref_txt.is_file():
        pytest.skip(
            f"reference voice sample not found under {VOICE_DIR} "
            "(set REFERENCE_VOICES_DIR to a directory containing eric/eric.wav and eric.txt)"
        )
    ref_text = ref_txt.read_text(encoding="utf-8").strip()
    wav_bytes = ref_wav.read_bytes()

    started = time.perf_counter()
    record = register_voice_bytes(client, wav_bytes, ref_text, name="tier1")
    upload_seconds = time.perf_counter() - started
    assert upload_seconds < 5.0  # spec SC-007

    runtime = components.readiness.runtime
    calls, restore = _spy_iter_audio_chunks(runtime)
    try:
        for cfg in (4.0, 1.0):
            before = len(calls)
            inline_response = client.post(
                "/v1/audio/speech",
                data={
                    "text": TEXT,
                    "instruction": INSTRUCTION,
                    "ref_text": ref_text,
                    "cfg_scale": str(cfg),
                    "seed": "42",
                    "max_new_tokens": str(MAX_NEW_TOKENS),
                },
                files={"ref_audio": ("eric.wav", wav_bytes, "audio/wav")},
            )
            assert inline_response.status_code == 200, inline_response.text
            inline_body = inline_response.content

            codes_samples, codes_accepted, codes_body = speak(
                client, events, voice_id=record["id"], text=TEXT, instruction=INSTRUCTION,
                cfg_scale=cfg, seed=42, ref_text=ref_text, max_new_tokens=MAX_NEW_TOKENS,
            )
            assert codes_accepted["reference"] == "voice_codes", codes_accepted
            assert len(codes_body) > 0
            assert np.any(codes_samples != 0), "audio is entirely silence"

            assert len(calls) == before + 2, (
                f"cfg_scale={cfg}: expected exactly 2 iter_audio_chunks calls "
                f"(inline, codes), got {len(calls) - before}"
            )
            inline_frames, codes_frames = calls[before], calls[before + 1]
            tokens_match = len(inline_frames) == len(codes_frames) and all(
                torch.equal(a, b) for a, b in zip(inline_frames, codes_frames)
            )
            assert tokens_match, (
                f"cfg_scale={cfg}: generated token frames diverged between the inline and "
                f"codes paths (inline {len(inline_frames)} frames, codes {len(codes_frames)} "
                "frames) -- unlike the PCM check below, this must be exact"
            )

            # Sanity only (module docstring): the runtime doesn't guarantee decode is
            # bit-reproducible call to call, so this doesn't compare inline_body to codes_body
            # at all -- only that each is a real, plausible response on its own.
            assert len(inline_body) == len(codes_body), (
                f"cfg_scale={cfg}: PCM length differs: inline {len(inline_body)} bytes, "
                f"codes {len(codes_body)} bytes"
            )
            for label, body in (("inline", inline_body), ("codes", codes_body)):
                assert len(body) > 0, f"cfg_scale={cfg}: {label} PCM is empty"
                pcm = np.frombuffer(body, dtype="<i2")
                assert np.any(pcm != 0), f"cfg_scale={cfg}: {label} PCM is entirely silence"
    finally:
        restore()
