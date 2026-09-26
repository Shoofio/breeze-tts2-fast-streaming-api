"""Standalone worker process for test_reference_encode_determinism.py.

Loads only the audio codec (no backbone model, no `MultiRequestStreamRuntime`) and sets
``torch.backends.cudnn.benchmark = True`` directly to reproduce the one condition that
matters for this bug: `--fast-all` leaves that flag on process-wide
(`models/stream_runtime/stream/runtime.py`'s `MultiRequestStreamRuntime.__init__`, verified
separately and cheaply by `_runtime_benchmark_flag_worker.py`). Building the real runtime here
too would cost ~85s per worker (a lazy `torch.compile` of every SnakeBeta module plus a CUDA
graph warmup, both irrelevant to what this worker checks) just to reach that same one-line
effect.

It then encodes one reference wav through `encode_prompt_waveform`, recording the cudnn flags
`encode_prompt_waveform` scopes the codec's own `encode` call to reach, and writes the codes and
those flags to disk.

Must run as a fresh, separate Python process: the bug this guards against is cuDNN's
``benchmark=True`` autotuning a different conv algorithm per *process* (the algorithm search
races candidate kernels against a wall clock, so which one "wins" isn't fixed within one
process, let alone across two) -- an in-process call can't exercise that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

from breeze_infer.audio import encode_prompt_waveform


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

    # The condition `--fast-all` creates process-wide (see module docstring): set it
    # directly rather than paying for a full runtime just to reach this one line.
    torch.backends.cudnn.benchmark = True

    seen_flags: dict[str, bool] = {}
    original_encode = audio_tokenizer.encode

    def recording_encode(*call_args: object, **call_kwargs: object):
        # Recorded *inside* the call so this reflects whatever encode_prompt_waveform's
        # cudnn scope has set at the moment the codec actually runs, not the ambient
        # process-wide state set above (which is what the bug used to leak through).
        seen_flags["benchmark"] = torch.backends.cudnn.benchmark
        seen_flags["deterministic"] = torch.backends.cudnn.deterministic
        return original_encode(*call_args, **call_kwargs)

    audio_tokenizer.encode = recording_encode

    wav, sample_rate = sf.read(args.wav, always_2d=True, dtype="float32")
    codes = encode_prompt_waveform(audio_tokenizer, wav, sample_rate)
    torch.save({"codes": codes, **seen_flags}, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
