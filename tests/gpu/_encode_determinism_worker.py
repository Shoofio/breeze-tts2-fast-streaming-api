"""Standalone worker process for test_reference_encode_determinism.py.

Loads only the audio codec (no backbone model), flips
``torch.backends.cudnn.benchmark`` on exactly the way the server does for
``--fast-all`` -- by building the real ``MultiRequestStreamRuntime`` the fast
codec path uses (``models/fast_streaming.py``'s ``FastBreezeStreamingRuntime._codec``
constructs this same object from just the codec, never the backbone, to decode
audio) -- then encodes one reference wav through ``encode_prompt_waveform`` and
writes the resulting codes to disk.

Must run as a fresh, separate Python process: the bug this guards against is
cuDNN's ``benchmark=True`` autotuning a different conv algorithm per *process*
(the algorithm search races candidate kernels against a wall clock, so which one
"wins" isn't fixed within one process, let alone across two) -- an in-process
call can't exercise that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

from breeze_infer.audio import encode_prompt_waveform
from models.stream_runtime import MultiRequestStreamRuntime, QwenStreamRuntimeConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSTokenizer

    device = torch.device("cuda:0")
    audio_tokenizer_dir = Path(args.ckpt_dir) / "audio_tokenizer"
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(audio_tokenizer_dir), device_map=str(device)
    )

    try:
        codec_dtype = next(audio_tokenizer.model.parameters()).dtype
    except StopIteration:
        codec_dtype = torch.float32

    # Reproduces models/fast_streaming.py's FastBreezeStreamingRuntime._codec():
    # the one production call that builds a fast (chunk_frames=1) codec runtime
    # from just the codec tokenizer. Its __init__
    # (models/stream_runtime/stream/runtime.py, MultiRequestStreamRuntime) is what
    # sets torch.backends.cudnn.benchmark = True for the whole process when
    # config.fast and device.type == "cuda" -- exactly the --fast-all condition.
    runtime_config = QwenStreamRuntimeConfig(
        chunk_frames=1,
        num_lanes=1,
        max_active_reqs=1,
        fast=True,
        device=device,
        dtype=codec_dtype,
    )
    MultiRequestStreamRuntime(audio_tokenizer, runtime_config)
    if not torch.backends.cudnn.benchmark:
        raise RuntimeError(
            "expected the fast codec runtime to enable cudnn.benchmark; it did not"
        )

    wav, sample_rate = sf.read(args.wav, always_2d=True, dtype="float32")
    codes = encode_prompt_waveform(audio_tokenizer, wav, sample_rate)
    torch.save(codes, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
