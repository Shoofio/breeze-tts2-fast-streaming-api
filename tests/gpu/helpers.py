"""Helpers shared by the GPU tests (fixtures live in conftest.py)."""

from __future__ import annotations

import numpy as np
import torch

from breeze_infer.runtime import set_all_seeds
from breeze_infer.templates import get_template, prepare_inputs, prepare_suffix_inputs


def synthesize(env, request: dict, *, cfg: float, seed: int, prefix=None, max_frames=None):
    """Run one request and return (audio float32, frames list)."""
    set_all_seeds(seed)
    if prefix is not None:
        inputs = prepare_suffix_inputs(
            env.tokenizer, env.model, request, guidance_scale=cfg
        )
    else:
        template = "ref_edit_tata" if request.get("ref_text") else "tts_instruction"
        inputs = prepare_inputs(
            env.tokenizer,
            env.model,
            [request],
            get_template(template),
            guidance_scale=cfg,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
    frames: list[list[int]] = []
    audio: list[np.ndarray] = []

    def observe(frame: torch.Tensor) -> None:
        frames.append(frame.detach().cpu().tolist())

    iterator = env.runtime.iter_audio_chunks(
        inputs,
        request_id=f"gpu-test-{seed}",
        seed=seed,
        token_observer=observe,
        prefix=prefix,
    )
    try:
        for chunk in iterator:
            audio.append(chunk.audio)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        iterator.close()
    return np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32), frames
