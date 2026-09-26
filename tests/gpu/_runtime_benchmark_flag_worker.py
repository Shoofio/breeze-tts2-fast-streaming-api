"""Standalone worker for the cheap `MultiRequestStreamRuntime` cudnn.benchmark assert
in test_reference_encode_determinism.py.

Constructs the *real* `MultiRequestStreamRuntime` (models/stream_runtime/stream/runtime.py)
with the same fast-codec config production uses (models/fast_streaming.py's
`FastBreezeStreamingRuntime._codec`), with two of its `__init__` steps patched to no-ops:

- `_validate_and_get_samples_per_code`: runs a real decoder forward pass, which is what
  actually triggers the lazy `torch.compile(..., mode="max-autotune-no-cudagraphs")` that
  `_compile_snakes` wrapped every SnakeBeta module's `forward` in -- the slowest single step
  in `__init__` by far.
- `_maybe_warmup_fast_codec`: captures CUDA graphs for each lane.

Both run strictly *after* the `torch.backends.cudnn.benchmark = True` line this worker is
checking, and neither touches `cudnn.benchmark` itself, so patching them out doesn't change
what's being verified -- only how long it takes to verify it: measured on this machine,
`MultiRequestStreamRuntime(...)` itself takes ~7 s with both patches applied (vs. the ~85 s the
unpatched steps cost). That's the actual saving; it's not the whole story for this worker's total
wall time, which is dominated by importing `qwen_tts`/`transformers`/`models.stream_runtime` in a
fresh process (~80 s measured here) plus loading the codec (~7 s) -- both unavoidable in any
fresh-process worker, this one included, and unrelated to what this file patches out. This is a
real `MultiRequestStreamRuntime` instance built from the real codec, not a source-level check or
a reimplementation of its `__init__` logic.

Runs in its own process, like `_encode_determinism_worker.py`, so a fully loaded codec and a
process-wide `cudnn.benchmark = True` can't leak into the pytest process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import mock

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSTokenizer

    from models.stream_runtime import MultiRequestStreamRuntime, QwenStreamRuntimeConfig

    device = torch.device("cuda:0")
    audio_tokenizer_dir = Path(args.ckpt_dir) / "audio_tokenizer"
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(audio_tokenizer_dir), device_map=str(device)
    )
    try:
        codec_dtype = next(audio_tokenizer.model.parameters()).dtype
    except StopIteration:
        codec_dtype = torch.float32

    # Matches models/fast_streaming.py's FastBreezeStreamingRuntime._codec() exactly:
    # chunk_frames=1, num_lanes=1, max_active_reqs=1, fast=True is the --fast-all shape.
    runtime_config = QwenStreamRuntimeConfig(
        chunk_frames=1,
        num_lanes=1,
        max_active_reqs=1,
        fast=True,
        device=device,
        dtype=codec_dtype,
    )
    with (
        mock.patch.object(
            MultiRequestStreamRuntime,
            "_validate_and_get_samples_per_code",
            return_value=1,
        ),
        mock.patch.object(MultiRequestStreamRuntime, "_maybe_warmup_fast_codec"),
    ):
        MultiRequestStreamRuntime(audio_tokenizer, runtime_config)

    if not torch.backends.cudnn.benchmark:
        raise RuntimeError(
            "MultiRequestStreamRuntime.__init__ did not set "
            "torch.backends.cudnn.benchmark=True for a fast codec on CUDA"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
