# ruff: noqa  (reference-only prototype; see README.md in this directory)
"""Shared bits for the Breeze MLX frame-loop prototype (measurement only)."""

import re
import sys
from pathlib import Path

import mlx.core as mx

W8 = Path.home() / ".cache/huggingface/hub/models--mlx-community--Breeze-TTS-2-mlx-8bit/snapshots/c6e4a2ff6ab9afba68b7853de802273ffe23fb49"
BF16 = Path.home() / ".cache/huggingface/hub/models--mlx-community--Breeze-TTS-2-mlx/snapshots/3c8829fb7fd335818f085cd2ef49b4100c0e46c8"
README = Path(__file__).resolve().parents[4] / "README.md"
PROTO = Path(__file__).resolve().parent

SENTENCE = "Hello there, this is a short test of the streaming voice."
INSTRUCTION = "A calm, warm voice."
CFG_SCALE = 4.0


def gate_passage() -> str:
    """Same text as the Phase-0 gate (README 'What this is' section)."""
    text = README.read_text()
    section = text.split("## What this is", 1)[1].split("## Requirements", 1)[0]
    section = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", section)
    section = re.sub(r"[*`>#]", "", section)
    return " ".join(section.split())


# Chosen to give roughly 30-45 s of speech and end on EOS (verified in RESULTS.md).
BENCH_PASSAGE = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. "
    "He trimmed the wick, polished the great lens, and wrote the weather in a worn leather book. "
    "Ships passed far out on the dark water, and none of them knew his name. "
    "Still, he liked to think that each captain glanced at the beam and felt a little safer. "
    "When the storm came in November, the waves rose higher than the rocks, "
    "and the old tower shook in the wind until morning."
)


def weights(label: str) -> Path:
    return {"8bit": W8, "bf16": BF16}[label]


def load_model(label: str):
    from mlx_audio.tts.utils import load
    return load(weights(label))


def text_by_name(name: str) -> str:
    return {"sentence": SENTENCE, "gate_passage": gate_passage(), "passage": BENCH_PASSAGE}[name]


def write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def make_fp32(model) -> None:
    """Diagnostic only: run backbone/depth/lm_head activations in float32 (quantized weights
    stay quantized). Changes this in-memory model instance, never the package."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    class _F32(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            return self.inner(x).astype(mx.float32)

    for mod in (model.backbone_model, model.depth_decoder, model.lm_head, model.text_encoder_proj):
        mod.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p,
                            mod.parameters()))
    model.backbone_model.embed_tokens.embed_audio_tokens = _F32(model.backbone_model.embed_tokens.embed_audio_tokens)
    model.depth_decoder.model.embed_tokens = _F32(model.depth_decoder.model.embed_tokens)
    orig = model._prompt_embeddings
    model._prompt_embeddings = lambda *a, **k: orig(*a, **k).astype(mx.float32)
