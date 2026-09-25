"""Single-request streaming inference for Breeze TTS 2."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from breeze_infer.audio import encode_prompt_waveform
from breeze_infer.compile_cache import MANIFEST_NAME, pin_torch_key, resolve_cache_dir
from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parent
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one WAV with Breeze TTS 2")
    parser.add_argument("model", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="Speak clearly and naturally.")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--compile-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError("--cfg-scale must be greater than 0")
    cache_dir = resolve_cache_dir(args.compile_cache_dir)
    pin_torch_key(cache_dir)

    has_ref_audio = args.ref_audio is not None
    has_ref_text = bool(args.ref_text and args.ref_text.strip())
    if has_ref_audio != has_ref_text:
        raise ValueError("--ref-audio and --ref-text must be provided together")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model,
        device=resolve_device(),
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )

    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(
            profile, manifest_path=cache_dir / MANIFEST_NAME
        )
        counters = manifest["compile_cache"]["counters"]
        print(
            f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms "
            f"(fx graph cache hits {counters['inductor.fxgraph_cache_hit']} "
            f"/ misses {counters['inductor.fxgraph_cache_miss']})"
        )

    request = {
        "id": "single-request",
        "text": args.text,
        "instruction": args.instruction,
        "speaker": "S0",
    }
    template_name = "tts_instruction"
    if args.ref_audio is not None:
        # templates.py only accepts pre-encoded reference codes (no path variant --
        # see breeze_infer/audio.py and R10), so the CLI encodes the file itself.
        try:
            wav, sample_rate = sf.read(args.ref_audio, always_2d=True, dtype="float32")
            request["ref_audio_codes"] = encode_prompt_waveform(
                audio_tokenizer, wav, sample_rate
            )
        # Narrow on purpose (review #10): a bad file (soundfile's own errors, plus
        # plain OSError for e.g. a permissions problem) or a codec that returned the
        # wrong shape (encode_prompt_waveform's ValueError) are the CLI's own
        # problem to explain with the file path attached. torch.OutOfMemoryError and
        # any other RuntimeError from the codec itself are not about this file and
        # must propagate with their real traceback intact.
        except (OSError, sf.SoundFileError, ValueError) as exc:
            raise ValueError(
                f"Could not encode reference audio '{args.ref_audio}': {exc}"
            ) from exc
        request["ref_text"] = args.ref_text.strip()
        template_name = "ref_edit_tata"

    set_all_seeds(args.seed)
    inputs = prepare_inputs(
        tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        args.output,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output_file:
        # The runtime's default length is now the model default (750 frames); the CLI keeps
        # its long-standing 1,500-frame cap by asking for it.
        for chunk in runtime.iter_audio_chunks(
            inputs,
            request_id="single-request",
            seed=args.seed,
            max_new_tokens=MAX_NEW_TOKENS,
        ):
            output_file.write(chunk.audio)

    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
