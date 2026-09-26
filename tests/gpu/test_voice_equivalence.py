"""Voice-store equivalence, ported from `api-alignment:tests/gpu/test_voice_equivalence.py`
(tasks.md T067).

The old test compared raw backbone logits between an inline reference and a continuation
(prefix) reference built from the same codes, using `FastBreezeStreamingRuntime` internals
(`_run_prefill`, `_build_branch_batch`, `_sampling_params`, ...) that predate this branch's
`templates.prepare_inputs`/`prepare_prefix_inputs`/`prepare_suffix_inputs` signature change
(no more `audio_tokenizer` argument) and its `_build_branch_batch(inputs, shape)` split. This
port keeps the outer statistical intent -- (d) in the old docstring: speaker identity is
statistically indistinguishable between the two paths, calibrated against genuine cross-voice
separation -- but drives both paths through the real server: a voice registered by
`POST /v1/voices` on a tmp directory, then spoken by `POST /v1/audio/speech` with `voice_id`.

`synthesis.voice_reference` (breeze_infer/synthesis.py) is the one rule for which path a
resolved voice takes: a `ref_text` override forces the codes path (`CodesRef`, the stored
codes inlined into the whole prompt every piece, like this branch's "inline"); with no
override it is the prefix path (`UnbuiltPrefix`, built once into cached backbone KV and
reused, like the old test's "continuation"). Passing the voice's own stored `ref_text` back
as the override therefore isolates exactly the one variable the old test varied (prefix vs.
inline), with everything else -- codes, text, instruction, cfg_scale, seed -- held fixed.

Speaker identity is measured the same way as the module it replaces: `encode_prompt_waveform`
re-encodes the response's PCM back into codec tokens (the same call `POST /v1/voices` uses to
encode a reference upload), and the codec's own quantizer decodes those tokens into the
latent the embedding is drawn from -- so this measures the same signal (the codec's own
representation of the speaker), just entered from audio instead of from raw frame ids.
"""

from __future__ import annotations

import asyncio
import io
import statistics
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from breeze_infer import api, bench_api
from breeze_infer.api import Components, create_app, load_in_background
from breeze_infer.audio import encode_prompt_waveform
from breeze_infer.gpu import GpuGate, GpuThread
from breeze_infer.limits import MAX_REF_SECONDS, MAX_REF_TEXT_CHARS
from breeze_infer.routes_health import Readiness
from breeze_infer.routes_speech import CpuTokenizer
from breeze_infer.runtime import resolve_device
from breeze_infer.settings import settings_from_args
from tests.fakes import RecordingEvents
from tests.gpu.test_speech_http import (
    EVENT_POLL_TIMEOUT_SECONDS,
    REFERENCE_VOICES_DIR,
    REPO_ROOT,
    SPEECH_PATH,
    TERMINAL_EVENTS,
    _shared_loaded_model,
    shut_down_cpu_tokenizer,
)

pytestmark = pytest.mark.gpu

VOICES_PATH = "/v1/voices"
# Never checked against a real startup scan here (every test opens a fresh, empty tmp
# directory), so any fixed value does: it only has to be the same value `POST /v1/voices`
# writes into a voice file and `open_voices` opens the directory with.
FINGERPRINT = "gpu-test-fingerprint"

CASES = [
    ("We need to discuss what happened last night.", "Speak slowly with a restrained, serious tone."),
    ("Thank you all for coming tonight.", "Gracious and warm."),
]
CFG = 4.0
MAX_NEW_TOKENS = 80
# research.md's own tolerances for this comparison (old test's (d)): how far the prefix path's
# mean similarity to the codes path may trail the codes path's own reseeded baseline, and how
# many more speaker outliers it may have than that baseline.
MEAN_SIMILARITY_SLACK = 0.02
OUTLIER_SLACK = 2
# The calibration check: cross-voice similarity must sit at least this far below same-voice
# similarity, or the embedding isn't discriminative enough to trust the comparison above.
DISCRIMINABILITY_MARGIN = 0.1


# --- Shared voice-app wiring (also used by test_voice_prefill_buckets.py and
# test_voice_tier1_equivalence.py) --------------------------------------------------------


def _voice_components(gpu_env, voices_dir: Path) -> tuple[Components, RecordingEvents]:
    """A `Components` wired to a real (empty) tmp voices directory: the same `open_voices`
    startup scan and `load_in_background` sequence `main()` runs, over the session's already
    warmed `gpu_env.runtime` and its shared CPU tokenizer copies
    (`test_speech_http._shared_loaded_model`) -- so `POST /v1/voices` here writes a real voice
    file and registers it exactly as production does, without reloading the model.
    """
    events = RecordingEvents()
    device = resolve_device()
    nonces = iter(range(1_000_000))
    components = Components(
        settings=settings_from_args([str(REPO_ROOT)]),
        events=events,
        gate=GpuGate(),
        gpu=GpuThread(device, torch.cuda.set_device),
        readiness=Readiness(),
        ws_port=lambda: 0,
        cpu_tokenizer=CpuTokenizer(),
        open_voices=partial(
            api.open_voices,
            voices_dir=voices_dir,
            events=events,
            codec_fingerprint=FINGERPRINT,
            now=lambda: datetime.now(timezone.utc),
            nonce=lambda: f"n{next(nonces)}",
        ),
    )
    loaded = _shared_loaded_model(gpu_env.runtime)
    server = SimpleNamespace(should_exit=False)
    ok = asyncio.run(load_in_background(components, lambda: loaded, server))
    assert ok, "load_in_background failed to open the tmp voices directory"
    return components, events


@pytest.fixture()
def voices_app(gpu_env, tmp_path) -> Iterator[tuple[TestClient, RecordingEvents, Components]]:
    """One app per test, its own tmp voices directory and its own gate/GPU-thread/events, as
    `tests/gpu/test_speech_http.py`'s own `speech_app` fixture does for the no-voices case."""
    components, events = _voice_components(gpu_env, tmp_path / "voices")
    client = TestClient(create_app(components))
    try:
        yield client, events, components
    finally:
        try:
            components.gpu.shutdown()
        finally:
            shut_down_cpu_tokenizer(components)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def register_voice_bytes(
    client: TestClient, wav_bytes: bytes, ref_text: str, *, name: str | None = None
) -> dict[str, Any]:
    """`POST /v1/voices` for a raw WAV payload and `ref_text`, and the parsed JSON record.

    Takes the exact bytes to upload (rather than an ndarray this helper would itself encode
    to WAV) so a caller that also sends those same bytes inline (`test_voice_tier1_equivalence`)
    is guaranteed byte-identical reference audio on both requests: encoding an ndarray to WAV a
    second time is not guaranteed to reproduce the first encode's samples exactly (a float32
    round trip through a 16-bit PCM file can shift a sample by 1 LSB), and `encode_prompt_waveform`
    is pinned deterministic (`tests/gpu/test_reference_encode_determinism.py`) only for identical
    *input samples* -- a 1-LSB-different reference changes the encoded reference codes, not just
    rounding noise downstream.
    """
    data = {"ref_text": ref_text}
    if name is not None:
        data["name"] = name
    response = client.post(
        VOICES_PATH, data=data, files={"ref_audio": ("clip.wav", wav_bytes, "audio/wav")}
    )
    assert response.status_code == 200, response.text
    return response.json()


def register_voice(
    client: TestClient,
    audio: np.ndarray,
    sample_rate: int,
    ref_text: str,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """`register_voice_bytes`, encoding `audio` to WAV itself. Fine whenever the caller has no
    independent copy of the exact upload bytes to keep in sync with (every test but
    `test_voice_tier1_equivalence`, which registers straight from a file's own bytes)."""
    return register_voice_bytes(client, _wav_bytes(audio, sample_rate), ref_text, name=name)


def _wait_for_new_terminal_event(events: RecordingEvents, start: int) -> None:
    """`test_speech_http._wait_for_terminal_event`, but scoped to the calls recorded from
    index `start` on: `events` here is shared by every request in a test (module docstring),
    so scanning the whole list would find an earlier request's own terminal event."""
    deadline = time.monotonic() + EVENT_POLL_TIMEOUT_SECONDS
    while not any(name in TERMINAL_EVENTS for name, _ in events.calls[start:]):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"no {TERMINAL_EVENTS} event within {EVENT_POLL_TIMEOUT_SECONDS}s "
                f"(got: {events.calls[start:]})"
            )
        time.sleep(0.01)


def speak(
    client: TestClient,
    events: RecordingEvents,
    *,
    voice_id: str,
    text: str,
    instruction: str | None = None,
    cfg_scale: float | None = None,
    seed: int,
    ref_text: str | None = None,
    max_new_tokens: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], bytes]:
    """`POST /v1/audio/speech` for a registered `voice_id`: `ref_text` forces the codes path
    (`synthesis.voice_reference`'s override rule); omitted, the request takes the prefix path.

    Returns (the response's PCM as float32 samples, the request's own `speech.accepted`
    fields, the raw PCM body).
    """
    data: dict[str, str] = {"text": text, "voice_id": voice_id, "seed": str(seed)}
    if instruction is not None:
        data["instruction"] = instruction
    if cfg_scale is not None:
        data["cfg_scale"] = str(cfg_scale)
    if ref_text is not None:
        data["ref_text"] = ref_text
    if max_new_tokens is not None:
        data["max_new_tokens"] = str(max_new_tokens)
    start = len(events.calls)
    response = client.post(SPEECH_PATH, data=data)
    assert response.status_code == 200, response.text
    _wait_for_new_terminal_event(events, start)
    new_calls = events.calls[start:]
    accepted = next(fields for name, fields in new_calls if name == "speech.accepted")
    completed = [fields for name, fields in new_calls if name == "speech.completed"]
    assert len(completed) == 1, f"expected exactly one speech.completed, got {new_calls}"
    body = response.content
    assert len(body) > 0 and len(body) % 2 == 0
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32767.0
    return samples, accepted, body


@torch.inference_mode()
def _speaker_embedding(gpu_env, audio: np.ndarray) -> torch.Tensor:
    """The same measure the old test used (its own `_speaker_embedding`/`_frames_to_codes`),
    entered from decoded PCM instead of raw generated frame ids: re-encode the audio into
    codec tokens (`encode_prompt_waveform`, the same call `POST /v1/voices` makes on an
    upload) and let the codec's own quantizer decode them back into the latent whose
    time-averaged, normalized vector is the embedding."""
    codes = encode_prompt_waveform(gpu_env.audio_tokenizer, audio, gpu_env.runtime.sample_rate)
    quantizer = gpu_env.audio_tokenizer.model.decoder.quantizer
    codes_btk = codes.long().T.unsqueeze(0).to(gpu_env.runtime.device)  # [1, codebooks, T]
    latent = quantizer.decode(codes_btk)  # [1, D, T]
    return torch.nn.functional.normalize(latent.float().mean(dim=-1)[0], dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


def _voice_sources(reference_clips) -> list[SimpleNamespace]:
    """The voices this test registers: two of `conftest.py`'s synthesized design clips, plus
    `bench_api.DEFAULT_REF_AUDIO` ("eric") and `$REFERENCE_VOICES_DIR/vale` when
    the checkout has them -- real recordings alongside the synthetic ones, so the cross-voice
    calibration below isn't only comparing designs against each other. Skips a real voice
    whose sample or transcript is missing, rather than failing the whole test over it."""
    sources = [
        SimpleNamespace(
            name=f"design-{clip.index}",
            audio=clip.audio,
            sample_rate=clip.sample_rate,
            ref_text=clip.ref_text,
        )
        for clip in reference_clips[:2]
    ]
    real_voices = [
        ("eric", bench_api.DEFAULT_REF_AUDIO),
        ("vale", REFERENCE_VOICES_DIR / "vale" / "vale.wav"),
    ]
    for name, wav_path in real_voices:
        txt_path = wav_path.with_suffix(".txt")
        if not (wav_path.is_file() and txt_path.is_file()):
            continue
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        sources.append(
            SimpleNamespace(
                name=name,
                audio=audio,
                sample_rate=int(sample_rate),
                ref_text=txt_path.read_text(encoding="utf-8").strip(),
            )
        )
    return sources


def test_prefix_path_matches_codes_path_for_the_same_voice(gpu_env, voices_app, reference_clips) -> None:
    client, events, _components = voices_app
    sources = _voice_sources(reference_clips)
    assert len(sources) >= 3, f"need at least 3 voices for cross-voice calibration, got {sources!r}"

    voices = [
        SimpleNamespace(source=source, record=register_voice(
            client, source.audio, source.sample_rate, source.ref_text, name=f"voice-{index}"
        ))
        for index, source in enumerate(sources)
    ]

    # codes_emb: the codes path (an override equal to the voice's own stored ref_text forces
    # it, `synthesis.voice_reference`), the closest new-API equivalent of the old "inline"
    # path. prefix_emb: no override, the cached-prefix path being validated. reseed_emb: the
    # codes path again, reseeded, the old test's own same-voice baseline (natural output
    # variation from a different seed alone, nothing path-related).
    codes_embeddings: dict[tuple[int, int], torch.Tensor] = {}
    per_case_similarity: list[tuple[float, float]] = []
    problems: list[str] = []

    for voice_index, voice in enumerate(voices):
        for case_index, (text, instruction) in enumerate(CASES):
            seed = 1000 * voice_index + case_index

            codes_samples, codes_accepted, _ = speak(
                client, events, voice_id=voice.record["id"], text=text, instruction=instruction,
                cfg_scale=CFG, seed=seed, ref_text=voice.source.ref_text, max_new_tokens=MAX_NEW_TOKENS,
            )
            assert codes_accepted["reference"] == "voice_codes", codes_accepted

            prefix_samples, prefix_accepted, _ = speak(
                client, events, voice_id=voice.record["id"], text=text, instruction=instruction,
                cfg_scale=CFG, seed=seed, max_new_tokens=MAX_NEW_TOKENS,
            )
            assert prefix_accepted["reference"] == "voice_prefix", prefix_accepted

            reseed_samples, reseed_accepted, _ = speak(
                client, events, voice_id=voice.record["id"], text=text, instruction=instruction,
                cfg_scale=CFG, seed=seed + 100_000, ref_text=voice.source.ref_text,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            assert reseed_accepted["reference"] == "voice_codes", reseed_accepted

            codes_emb = _speaker_embedding(gpu_env, codes_samples)
            prefix_emb = _speaker_embedding(gpu_env, prefix_samples)
            reseed_emb = _speaker_embedding(gpu_env, reseed_samples)
            codes_embeddings[(voice_index, case_index)] = codes_emb

            per_case_similarity.append((_cosine(codes_emb, prefix_emb), _cosine(codes_emb, reseed_emb)))

    # Calibration: different voices, same case, must be clearly less similar than the
    # same-voice reseed baseline -- otherwise the embedding can't tell voices apart at all,
    # and the comparison above proves nothing.
    cross_voice = [
        _cosine(codes_embeddings[(a, case_index)], codes_embeddings[(b, case_index)])
        for case_index in range(len(CASES))
        for a in range(len(voices))
        for b in range(a + 1, len(voices))
    ]
    same_voice_reseed = [reseed_sim for _prefix_sim, reseed_sim in per_case_similarity]
    mean_prefix = statistics.mean(sim for sim, _reseed_sim in per_case_similarity)
    mean_same = statistics.mean(same_voice_reseed)
    mean_cross = statistics.mean(cross_voice)
    midpoint = (mean_same + mean_cross) / 2
    prefix_below = [
        (index, round(sim, 3))
        for index, (sim, _reseed_sim) in enumerate(per_case_similarity)
        if sim <= midpoint
    ]
    reseed_below = [
        (index, round(reseed_sim, 3))
        for index, (_sim, reseed_sim) in enumerate(per_case_similarity)
        if reseed_sim <= midpoint
    ]
    print(
        f"\nvoices={len(voices)} cases={len(CASES)}; "
        f"prefix-vs-codes sim {mean_prefix:.3f}; same-voice reseed sim {mean_same:.3f}; "
        f"cross-voice sim {mean_cross:.3f}; midpoint {midpoint:.3f}; "
        f"below midpoint: prefix {prefix_below}, reseed {reseed_below}"
    )
    assert not problems, "\n".join(problems)
    assert mean_cross <= mean_same - DISCRIMINABILITY_MARGIN, (
        "similarity measure is not discriminative: "
        f"cross-voice {mean_cross:.3f} vs same-voice {mean_same:.3f}"
    )
    assert mean_prefix >= mean_same - MEAN_SIMILARITY_SLACK, (
        f"prefix-vs-codes similarity {mean_prefix:.3f} below reseed baseline {mean_same:.3f}"
    )
    assert len(prefix_below) <= len(reseed_below) + OUTLIER_SLACK, (
        f"more speaker outliers than a reseeded codes-path run: prefix {prefix_below} vs reseed {reseed_below}"
    )


def test_prefix_over_548_tokens_now_builds_and_synthesizes(gpu_env, voices_app, reference_clips) -> None:
    """research.md R12 point 4: the old guard (`P:fast_streaming.py:864-867`) rejected any
    voice prefix over 548 tokens outright. The new one (`build_reference_prefix`,
    `MIN_SUFFIX_ROOM`-based) only refuses a prefix that would leave no room for a short
    default-instruction suffix, so a longer prefix is now accepted, builds and speaks.

    The reference is one of `conftest.py`'s generated clips, repeated until its audio (capped
    at `MAX_REF_SECONDS`) and its repeated transcript push the registered voice's measured
    `prefix_len` past the old boundary. The audio alone saturates at `MAX_REF_SECONDS` (about
    375 frames for this codec) well before 6 repeats of the clip are even needed, so the text
    is repeated far more than the audio (independently, up to `MAX_REF_TEXT_CHARS`) to supply
    the rest of the needed prefix length -- a first attempt that repeated both the same number
    of times only reached `prefix_len=492` (375 audio frames + 117 text tokens), short of 548.
    """
    client, events, components = voices_app
    clip = reference_clips[0]
    audio_repeats = 6
    text_repeats = 20
    long_audio = np.concatenate([clip.audio] * audio_repeats)
    max_samples = int(MAX_REF_SECONDS * clip.sample_rate)
    long_audio = long_audio[:max_samples]
    long_ref_text = (" ".join([clip.ref_text] * text_repeats))[:MAX_REF_TEXT_CHARS]

    record = register_voice(client, long_audio, clip.sample_rate, long_ref_text, name="long_prefix_voice")
    voice_id = record["id"]

    resolved = components.voices.get().registry.lookup(voice_id)
    assert resolved is not None
    print(f"\nregistered prefix_len={resolved.prefix_len} frames={record['frames']}")
    assert resolved.prefix_len > 548, (
        f"prefix_len={resolved.prefix_len} did not exceed the old 548-token guard "
        f"(frames={record['frames']}, ref_text chars={len(long_ref_text)}); "
        "grow the repeated reference further"
    )

    samples, accepted, body = speak(
        client, events, voice_id=voice_id,
        text="A quick check that this long voice still speaks.",
        seed=7, max_new_tokens=60,
    )
    assert accepted["reference"] == "voice_prefix", accepted
    assert len(body) > 0 and len(body) % 2 == 0
    assert np.any(samples != 0), "audio is entirely silence"
