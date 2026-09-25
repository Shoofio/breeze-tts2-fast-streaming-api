"""Load and warm up the streaming runtime from `Settings` (one job: model loading).

Everything here runs on the `GpuThread` (research.md R14), in the background, while `/health`
answers `503 loading`. Nothing prints: the composition root turns the returned report into the
`model.loaded` event.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from breeze_infer.compile_cache import MANIFEST_NAME, pin_torch_key, resolve_cache_dir
from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
from breeze_infer.settings import Settings
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

FAST_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fast.json"

# The static context the CUDA graphs are captured for (BC-47). A property of the runtime, not a
# request limit, so it lives here rather than in limits.py.
MAX_SEQ_LEN = 2048


@dataclass(frozen=True)
class LoadedModel:
    """The ready runtime plus the facts worth reporting in `model.loaded`."""

    runtime: Any
    report: dict[str, Any]


def configure_compile_cache(
    cli_value: Path | None, environ: MutableMapping[str, str]
) -> tuple[Path, str]:
    """Choose and export the Inductor cache dir, then pin torch's source key.

    Must run before the first `torch.compile`. `environ` is the process environment, passed in
    by the composition root: torch reads `TORCHINDUCTOR_CACHE_DIR` from it, so the choice has to
    be written back there. Returns the directory and the torch-key status (`hit`/`miss`/...).
    """
    cache_dir = resolve_cache_dir(cli_value, environ=environ)
    return cache_dir, pin_torch_key(cache_dir)


def streaming_config(settings: Settings) -> FastStreamingConfig:
    """Map the launch flags onto the runtime's configuration."""
    return FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS_CEILING,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
    )


def warmup_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Total warmup time plus compile-cache hit/miss counts, from a warmup manifest."""
    counters = manifest.get("compile_cache", {}).get("counters", {})
    return {
        "warmup_ms": round(float(manifest["total_elapsed_ms"]), 2),
        "fx_graph_cache_hits": counters.get("inductor.fxgraph_cache_hit", 0),
        "fx_graph_cache_misses": counters.get("inductor.fxgraph_cache_miss", 0),
    }


def load_model(
    settings: Settings, device: str, environ: MutableMapping[str, str]
) -> LoadedModel:
    """Configure the compile cache, load the checkpoint, and warm up the fast paths."""
    cache_dir, torch_key = configure_compile_cache(settings.compile_cache_dir, environ)
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model_path,
        device=device,
        attn_implementation=settings.attn_implementation,
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, streaming_config(settings), tokenizer=tokenizer
    )

    report: dict[str, Any] = {
        "device": device,
        "compile_cache_dir": str(cache_dir),
        "torch_key": torch_key,
        "warmup_ms": None,
    }
    if runtime.fast_enabled:
        profile = replace(
            load_warmup_profile(FAST_CONFIG), codec_chunk_frames=runtime.codec_chunk_frames
        )
        manifest = runtime.warmup_from_profile(
            profile, manifest_path=cache_dir / MANIFEST_NAME
        )
        report.update(warmup_report(manifest))
    return LoadedModel(runtime=runtime, report=report)
