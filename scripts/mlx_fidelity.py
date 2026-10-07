"""How far a quantized MLX checkpoint drifts from bf16, per stage, on the same tokens.

bf16 generates each line (its own sampling, seed 42); every variant is fed exactly the same codes in lockstep, and at
every step the two models' guided logits (what the samplers see) are compared: KL(bf16 || variant) and whether the
top token agrees. "backbone" is codebook 0 (prosody and timing, where a voice direction lands); "depth" is codebooks
1-15 (acoustic detail). Sampling noise drops out, so small differences between quantizations show.

    python scripts/mlx_fidelity.py --ref-audio REF.wav --ref-text "..." 8bit=mxfp8 mixed=depth:8 int8=depth:8,backbone:8

A variant is NAME=mxfp8 (the community 8-bit checkpoint) or NAME=<BREEZE_MLX_QUANT> applied to the bf16 checkpoint.
Needs both checkpoints in the HuggingFace cache (README, macOS quick start) and room for two models at once (~16 GB).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from bench_mac import lines

from breeze_infer.audio import encode_prompt_waveform
from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.synthesis import CodesRef, prepare_piece
from models.fast_streaming import select_fast_cfg
from models.mlx_streaming import (
    _CONDITIONAL,
    _NEGATIVE,
    MlxAudioTokenizer,
    MlxSpeedOptions,
    _Generation,
    _sample,
    _to_mlx_prompt,
    load_mlx_runtime,
)

HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
BF16 = "models--mlx-community--Breeze-TTS-2-mlx"
MXFP8 = "models--mlx-community--Breeze-TTS-2-mlx-8bit"
MAX_FRAMES = 160


def snapshot(repo: str) -> Path:
    found = sorted(glob.glob(str(HUB / repo / "snapshots" / "*")))
    if not found:
        sys.exit(f"{repo} is not in {HUB}: download it first (README, macOS quick start)")
    return Path(found[-1])


def load(spec: str):
    if spec == "mxfp8":
        return load_mlx_runtime(snapshot(MXFP8), MlxSpeedOptions())
    return load_mlx_runtime(snapshot(BF16), MlxSpeedOptions(quantize="" if spec == "bf16" else spec))


def generation(runtime, inputs) -> _Generation:
    cfg = select_fast_cfg(inputs)
    rows = [_CONDITIONAL, _NEGATIVE] if cfg.mode == "single_cfg" else [_CONDITIONAL]
    model = runtime._mlx_model
    return _Generation(
        model,
        [_to_mlx_prompt(inputs, keys, model.config) for keys in rows],
        frames=MAX_FRAMES + 1,
        guidance=cfg.guidance_scale,
        backbone_sampling=runtime._backbone_sampling,
        depth_sampling=runtime._depth_sampling,
        repetition_penalty=runtime.config.repetition_penalty,
        seed=42,
    )


def kl_top(p_logits, q_logits):
    """KL(p || q) in nats over p's finite entries, and whether the argmaxes agree."""
    lp = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    lq = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
    finite = mx.isfinite(lp)
    kl = mx.sum(mx.where(finite, mx.exp(lp) * (lp - mx.where(mx.isfinite(lq), lq, -1e4)), 0.0), axis=-1)
    return kl, mx.argmax(p_logits, axis=-1) == mx.argmax(q_logits, axis=-1)


def compare(ref_rt, var_rt, inputs) -> dict[str, list]:
    gr, gv = generation(ref_rt, inputs), generation(var_rt, inputs)
    eos = ref_rt._mlx_model.vocab_size
    last = ref_rt._mlx_model.num_codebooks - 1
    stats: dict[str, list] = {"bb_kl": [], "bb_top": [], "dp_kl": [], "dp_top": []}
    for _ in range(MAX_FRAMES):
        lr, lv = gr.backbone_logits(), gv.backbone_logits()
        first, _ = _sample(lr, gr._next_key(), gr._backbone_sampling)
        gv._next_key()
        kl, top = kl_top(lr, lv)
        mx.eval(first, kl, top)
        stats["bb_kl"].append(kl.item())
        stats["bb_top"].append(bool(top.item()))
        if int(first.item()) == eos:
            break
        sr, sv = gr.depth_steps(first), gv.depth_steps(first)
        codes, dkl, dtop = [first], [], []
        for codebook in range(1, last + 1):
            a, b = sr.logits(codebook), sv.logits(codebook)
            code, _ = _sample(a, gr._next_key(), gr._depth_sampling)
            gv._next_key()
            k, t = kl_top(a, b)
            dkl.append(k)
            dtop.append(t)
            codes.append(code)
            if codebook < last:
                sr.feed(code, codebook)
                sv.feed(code, codebook)
        frame = mx.concatenate(codes).astype(mx.int32)
        dkl, dtop = mx.concatenate(dkl), mx.concatenate(dtop)
        mx.eval(frame, dkl, dtop)
        stats["dp_kl"] += np.array(dkl).tolist()
        stats["dp_top"] += np.array(dtop).tolist()
        gr.advance(frame)
        gv.advance(frame)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("variants", nargs="+", help="NAME=mxfp8 or NAME=<BREEZE_MLX_QUANT>, e.g. mixed=depth:8")
    a = ap.parse_args()
    ref_rt = load("bf16")
    wav, sr = sf.read(str(a.ref_audio), dtype="float32")
    reference = CodesRef(codes=encode_prompt_waveform(MlxAudioTokenizer(ref_rt._codec), wav, sr), ref_text=a.ref_text)
    prepared = [
        ("directed" if line["instruction"] else "plain",
         prepare_piece(ref_rt.tokenizer, ref_rt.model, reference, line["text"],
                       line["instruction"] or DEFAULT_INSTRUCTION, a.cfg if line["instruction"] else 1.0))
        for line in lines()
    ]
    for item in a.variants:
        name, _, spec = item.partition("=")
        var_rt = load(spec)
        totals: dict[str, dict[str, list]] = {}
        for mode, inputs in prepared:
            for key, values in compare(ref_rt, var_rt, inputs).items():
                totals.setdefault(mode, {}).setdefault(key, []).extend(values)
        for mode, t in totals.items():
            print(f"{name:8s} {spec:20s} {mode:8s} backbone KL {np.mean(t['bb_kl']):.4f} top-1 {100 * np.mean(t['bb_top']):5.1f} % | "
                  f"depth KL {np.mean(t['dp_kl']):.4f} top-1 {100 * np.mean(t['dp_top']):5.1f} % | {len(t['bb_kl'])} frames", flush=True)
        del var_rt
        mx.clear_cache()


if __name__ == "__main__":
    main()
