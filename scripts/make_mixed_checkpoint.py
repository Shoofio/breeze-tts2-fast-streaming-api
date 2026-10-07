"""Save the `mixed` model as its own MLX checkpoint: bf16, with the depth decoder's linear layers in int8.

The same model `--precision mixed` builds at load (`BREEZE_MLX_QUANT=depth:8`, `quantize_parts`), saved so that stock
mlx-audio and this server load it as is: config.json gets an affine 8-bit "quantization" entry, and mlx-audio's loader
quantizes exactly the layers whose weights come with scales (the depth decoder's), leaving the rest bf16.

    python scripts/make_mixed_checkpoint.py --out DIR [--src BF16_SNAPSHOT] [--group 64]

--src defaults to the pinned bf16 snapshot in the HuggingFace cache (README, macOS quick start). The tokenizer, the
codec (audio_tokenizer/), LICENSE and NOTICE are copied unchanged; the weights stay under the BreezeBlue Research and
Non-Commercial License. After saving, the checkpoint is loaded back with stock mlx-audio and every tensor is compared
with the in-memory conversion; a difference stops with an error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from mlx_audio.tts.utils import load

from models.mlx_streaming import MlxSpeedOptions, quantize_parts

BF16_REPO = "models--mlx-community--Breeze-TTS-2-mlx"
BF16_REVISION = "3c8829fb7fd335818f085cd2ef49b4100c0e46c8"
SHARD_BYTES = 5 * 1024**3
SKIPPED_FILES = {"config.json", "README.md", "model.safetensors.index.json", ".gitattributes"}


def default_src() -> Path:
    import os

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return home / "hub" / BF16_REPO / "snapshots" / BF16_REVISION


def original_keys(src: Path) -> set[str]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    return set(mx.load(str(src / "model.safetensors")))


def save_weights(out: Path, weights: dict[str, mx.array]) -> None:
    shards: list[dict[str, mx.array]] = [{}]
    size = 0
    for key in sorted(weights):
        if size + weights[key].nbytes > SHARD_BYTES and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][key] = weights[key]
        size += weights[key].nbytes
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else "model.safetensors"
        mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
        weight_map.update(dict.fromkeys(shard, name))
    index = {"metadata": {"total_size": sum(w.nbytes for w in weights.values())}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--group", type=int, default=64)
    a = ap.parse_args()
    src = a.src or default_src()
    if not (src / "config.json").exists():
        sys.exit(f"no bf16 checkpoint at {src}: download it first (README, macOS quick start)")
    a.out.mkdir(parents=True, exist_ok=False)

    speed = MlxSpeedOptions(quantize="depth:8", group_size=a.group)
    model = load(src)
    quantize_parts(model, speed)
    params = dict(tree_flatten(model.parameters()))
    keep = original_keys(src)
    quantized = {k for k in params if k.endswith((".scales", ".biases")) and k.rsplit(".", 1)[0] + ".weight" in keep}
    weights = {k: v for k, v in params.items() if k in keep or k in quantized}
    missing = keep - set(weights)
    if missing:
        sys.exit(f"the model lacks {len(missing)} of the checkpoint's tensors, e.g. {sorted(missing)[:3]}")
    save_weights(a.out, weights)

    config = json.loads((src / "config.json").read_text())
    config["quantization"] = config["quantization_config"] = {"group_size": a.group, "bits": 8, "mode": "affine"}
    (a.out / "config.json").write_text(json.dumps(config, indent=2))
    for item in src.iterdir():
        if item.name in SKIPPED_FILES or item.name.startswith("model") and item.suffix == ".safetensors":
            continue
        if item.is_dir():
            shutil.copytree(item, a.out / item.name)
        else:
            shutil.copy2(item, a.out / item.name)
    print(f"saved {len(weights)} tensors ({len(quantized) // 2} quantized layers) to {a.out}", flush=True)

    # Load it back the stock way and compare every tensor with the in-memory conversion.
    again = load(a.out)
    reloaded = dict(tree_flatten(again.parameters()))
    if set(reloaded) != set(params):
        sys.exit(f"reloaded tensors differ in names: {sorted(set(reloaded) ^ set(params))[:5]}")
    for key, value in params.items():
        if value.dtype != reloaded[key].dtype or not mx.array_equal(value, reloaded[key]).item():
            sys.exit(f"reloaded tensor differs: {key}")
    layers = dict(again.named_modules())
    q = sum(isinstance(m, nn.QuantizedLinear) for k, m in layers.items() if k.startswith("depth_decoder"))
    b = sum(isinstance(m, nn.QuantizedLinear) for k, m in layers.items() if k.startswith("backbone_model"))
    if q == 0 or b != 0:
        sys.exit(f"unexpected quantization after reload: depth decoder {q} quantized layers, backbone {b}")
    print(f"verified: stock mlx-audio loads it identically ({len(params)} tensors; depth decoder {q} int8 layers, backbone bf16)")


if __name__ == "__main__":
    main()
