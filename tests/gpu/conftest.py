"""Shared GPU fixtures: one loaded, warmed fast-path runtime per session."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from breeze_infer.audio import encode_prompt_waveform
from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    update_generation_config_for_breeze,
)
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile
from tests.gpu.helpers import synthesize

REPO_ROOT = Path(__file__).resolve().parents[2]

VOICE_DESIGNS = [
    "A calm adult male voice, clear and natural.",
    "A bright, energetic young woman with a quick delivery.",
    "A deep, slow, gravelly older man.",
    "A soft-spoken woman with a gentle, warm tone.",
    "A crisp, formal male newsreader voice.",
]
REFERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant hills."
)


@pytest.fixture(scope="session")
def gpu_env(breeze_model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    tokenizer, model, audio_tokenizer = load_runtime(
        breeze_model, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    config = FastStreamingConfig(
        max_new_tokens=1500,
        max_seq_len=2048,
        fast_all=True,
        collect_timing=True,
        repetition_penalty=1.1,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    profile = load_warmup_profile(REPO_ROOT / "configs" / "fast.json")
    profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
    manifest = runtime.warmup_from_profile(profile)
    return SimpleNamespace(
        tokenizer=tokenizer,
        model=model,
        audio_tokenizer=audio_tokenizer,
        runtime=runtime,
        manifest=manifest,
    )


@pytest.fixture(scope="session")
def reference_clips(gpu_env):
    """Five model-generated reference clips with their codes and transcript."""
    clips = []
    for index, design in enumerate(VOICE_DESIGNS):
        request = {
            "id": f"design-{index}",
            "text": REFERENCE_TEXT,
            "instruction": design,
            "speaker": "S0",
        }
        audio, _ = synthesize(gpu_env, request, cfg=4.0, seed=100 + index)
        codes = encode_prompt_waveform(
            gpu_env.audio_tokenizer, audio, gpu_env.runtime.sample_rate
        )
        clips.append(
            SimpleNamespace(
                index=index,
                audio=audio,
                codes=codes,
                ref_text=REFERENCE_TEXT,
                sample_rate=gpu_env.runtime.sample_rate,
            )
        )
    return clips
