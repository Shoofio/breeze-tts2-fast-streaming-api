"""Load and warm up the streaming runtime from `Settings` (one job: model loading).

Everything here runs on the `GpuThread` (research.md R14), in the background, while `/health`
answers `503 loading`. Nothing prints: the composition root turns the returned report into the
`model.loaded` event.
"""

from __future__ import annotations

import copy
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
    """The ready runtime plus the facts worth reporting in `model.loaded`.

    `cpu_tokenizer` and `sizing_tokenizer` are two copies of the runtime's tokenizer, for the
    speech route's two CPU workers (`routes_speech.CpuTokenizer`): the room check before the
    gate, and the lease holder's anchor sizing. Neither may share the GPU thread's tokenizer,
    nor each other, since all three threads can tokenize at once. Required: the speech route
    cannot size a request without them (`api.Components.mark_ready` installs them together
    with the runtime). `from_runtime` makes the copies.
    """

    runtime: Any
    report: dict[str, Any]
    cpu_tokenizer: Any
    sizing_tokenizer: Any

    @classmethod
    def from_runtime(cls, runtime: Any, report: dict[str, Any]) -> LoadedModel:
        """`runtime` with its own tokenizer copied twice, once for each CPU worker.

        Call it on the GPU thread before the server reports ready, while nothing else uses the
        tokenizer: a deep copy of a real one takes about 700 ms, which would otherwise stall
        the event loop (`CpuTokenizer.install`, from `mark_ready`) or, made lazily, the first
        speech request.
        """
        return cls(
            runtime=runtime,
            report=report,
            cpu_tokenizer=copy.deepcopy(runtime.tokenizer),
            sizing_tokenizer=copy.deepcopy(runtime.tokenizer),
        )


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
    """Load the checkpoint for the chosen backend and warm it up before the server reports ready."""
    if settings.backend == "mlx":
        return _load_mlx_model(settings, device)
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
        "backend": "cuda",
        "weights": "bf16",
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
    # On the GPU thread, before the server reports ready (`LoadedModel.from_runtime`).
    return LoadedModel.from_runtime(runtime, report)


def _load_mlx_model(settings: Settings, device: str) -> LoadedModel:
    """Load the MLX checkpoint and compile its Metal kernels before the first request.

    Imported here, not at the top: `models.mlx_streaming` needs mlx, which only a Mac has, and
    this module must still import on Linux and Windows. There is no compile cache or torch key
    on MLX, so those report fields are null.
    """
    from models.mlx_streaming import load_mlx_runtime

    runtime = load_mlx_runtime(settings.model_path)
    warmup_ms = runtime.warmup()
    report: dict[str, Any] = {
        "backend": "mlx",
        "weights": settings.weights,
        "device": device,
        "compile_cache_dir": None,
        "torch_key": None,
        "warmup_ms": round(warmup_ms, 2),
        "fx_graph_cache_hits": None,
        "fx_graph_cache_misses": None,
    }
    return LoadedModel.from_runtime(runtime, report)
