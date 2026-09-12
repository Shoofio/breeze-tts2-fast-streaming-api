"""Persist ``torch.compile`` artifacts across server restarts.

CUDA graphs are bound to live device memory and must be recaptured every
process start. The ``torch.compile`` work that runs before capture (Triton
codegen, autotuning, AOTAutograd) is cacheable, but Inductor's default cache
lives in the system temp dir, which Ubuntu and WSL clear at boot. This module
pins that cache to a persistent location and reports hit/miss counters so a
warm start can be verified from the warmup manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE_DIR_ENV = "TORCHINDUCTOR_CACHE_DIR"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "torchinductor"
MANIFEST_NAME = "warmup_manifest.json"
TORCH_KEY_FILE = "torch_key.json"
_SOURCE_SUFFIXES = (".py", ".cpp", ".h", ".ld")
_torch_key_status = "not-attempted"

_COUNTER_KEYS: dict[str, tuple[str, ...]] = {
    "inductor": ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass"),
    "aot_autograd": (
        "autograd_cache_hit",
        "autograd_cache_miss",
        "autograd_cache_bypass",
        "autograd_cache_guard_miss",
    ),
    "stats": ("unique_graphs", "calls_captured"),
}


def resolve_cache_dir(
    cli_value: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    default: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Pick the Inductor cache directory and export it for torch.

    Precedence: explicit CLI value, then an already-set ``TORCHINDUCTOR_CACHE_DIR``
    (the Windows launcher and user shells set one), then the repo-local default.
    Must run before the first ``torch.compile`` call: torch reads the variable
    lazily but freezes the result on first use.
    """
    env = os.environ if environ is None else environ
    if cli_value is not None:
        chosen = Path(cli_value).expanduser().resolve()
    elif env.get(CACHE_DIR_ENV):
        chosen = Path(env[CACHE_DIR_ENV]).expanduser().resolve()
    else:
        chosen = default
    chosen.mkdir(parents=True, exist_ok=True)
    env[CACHE_DIR_ENV] = str(chosen)
    return chosen


def inductor_source_fingerprint() -> str:
    """Cheap stat-only digest of torch's inductor source tree.

    Reading every file (what ``torch_key`` does) costs seconds on a slow or
    cold filesystem; sizes and mtimes change on any reinstall, which is the
    only event that should invalidate the pinned key.
    """
    import torch
    import torch._inductor

    root = Path(torch._inductor.__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(torch.__version__.encode("utf-8"))
    digest.update(str(root).encode("utf-8"))
    for path in sorted(root.rglob("*")):
        if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        )
    return digest.hexdigest()


def pin_torch_key(
    cache_dir: Path,
    *,
    torch_key: Callable[[], bytes] | None = None,
    fingerprint: str | None = None,
) -> str:
    """Reuse torch's inductor source hash across process starts.

    Inductor hashes its own source tree once per process to key every cache
    entry. The hook ``torch_key.set`` lets a caller prepopulate that value, so
    it is stored next to the cache with a stat-based fingerprint and reused
    while the install is unchanged. Returns ``hit``, ``miss`` (computed and
    saved), or ``unavailable``. Must run before the first ``torch.compile``.
    """
    global _torch_key_status
    if torch_key is None:
        try:
            from torch._inductor.codecache import torch_key as torch_key_fn
        except ImportError:  # pragma: no cover - torch missing
            _torch_key_status = "unavailable"
            return _torch_key_status
        torch_key = torch_key_fn
    setter = getattr(torch_key, "set", None)
    if setter is None:
        _torch_key_status = "unavailable"
        return _torch_key_status
    if fingerprint is None:
        fingerprint = inductor_source_fingerprint()
    record = Path(cache_dir) / TORCH_KEY_FILE
    try:
        saved = json.loads(record.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fingerprint:
            setter(bytes.fromhex(saved["key"]))
            _torch_key_status = "hit"
            return _torch_key_status
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        pass
    key = torch_key()
    record.write_text(
        json.dumps({"fingerprint": fingerprint, "key": key.hex()}, indent=2) + "\n",
        encoding="utf-8",
    )
    _torch_key_status = "miss"
    return _torch_key_status


def compile_cache_stats() -> dict[str, int]:
    """Snapshot torch's compile cache counters as a flat, JSON-friendly dict.

    Missing counters read as zero so the shape is stable whether or not any
    compile has happened (or torch is even importable).
    """
    try:
        from torch._dynamo.utils import counters
    except ImportError:  # pragma: no cover - torch missing or too old
        counters = {}
    stats: dict[str, int] = {}
    for group, keys in _COUNTER_KEYS.items():
        bucket = counters.get(group, {}) if hasattr(counters, "get") else {}
        for key in keys:
            stats[f"{group}.{key}"] = int(bucket.get(key, 0))
    return stats


def diff_stats(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Counter deltas between two ``compile_cache_stats`` snapshots."""
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def compile_phase_seconds() -> dict[str, float]:
    """Aggregate torch's per-phase compile timers (dynamo tracing, AOT, inductor)."""
    try:
        from torch._dynamo.utils import compilation_time_metrics
    except ImportError:  # pragma: no cover - torch missing or too old
        return {}
    return {
        name: round(sum(values), 3)
        for name, values in compilation_time_metrics.items()
        if values
    }


def describe(cache_dir: Path | None = None) -> dict[str, Any]:
    """Manifest block describing the active cache configuration."""
    active = os.environ.get(CACHE_DIR_ENV)
    return {
        "cache_dir": str(cache_dir) if cache_dir is not None else active,
        "torch_key": _torch_key_status,
        "counters": compile_cache_stats(),
        "phase_seconds": compile_phase_seconds(),
    }
