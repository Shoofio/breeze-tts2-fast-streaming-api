"""GPU regression test for cross-process reference-encode determinism.

Finding (2026-09-25, user-approved deviation -- see
specs/003-cpp-compatible-api/research.md R18): with ``--fast-all``,
``torch.backends.cudnn.benchmark = True`` is set for the whole server process
(``models/stream_runtime/stream/runtime.py``'s ``MultiRequestStreamRuntime.__init__``).
cuDNN then autotunes whichever conv algorithm looks fastest *in that process* and
reuses it, and separate processes can autotune to different algorithms. Those
algorithms round differently, so about 1% of the fine-codebook codes (codebooks
6-15) come out different for the *same* reference wav across a server restart --
same wav, same text, same seed, different audio. ``breeze_infer/audio.py``'s
``encode_prompt_waveform`` now scopes ``benchmark=False, deterministic=True``
around just the codec's ``encode`` call to fix this.

This test does not merely check that two fresh-process encodes happen to agree: without
the fix, cuDNN's autotuner often (not always) picks the same algorithm across processes
anyway, so a bare "codes are equal" assertion can pass on a lucky run even with the bug
present. Instead each worker records the cudnn flags actually in effect *during* the
codec's own encode call, and the test asserts those directly (``benchmark=False,
deterministic=True``) in addition to the codes matching -- that assertion fails
deterministically, every run, whenever the fix regresses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from breeze_infer.bench_api import DEFAULT_REF_AUDIO

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
# bench_api derives its default reference wav from REFERENCE_VOICES_DIR (None when unset),
# so reuse it as the single source of truth for "the bench reference wav".
REFERENCE_WAV = DEFAULT_REF_AUDIO

# Three, not two: guards against a fix that happens to survive one lucky pair but not a
# third process.
NUM_WORKERS = 3


def _run_worker(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    # Run as `-m tests.gpu.<module>` with cwd=repo root (not a bare script path) so the
    # worker's `breeze_infer`/`models` imports resolve the same way they do under
    # `python -m pytest` from the repo root.
    return subprocess.run(
        [sys.executable, "-m", f"tests.gpu.{module}", *args],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _fail_worker(name: str, result: subprocess.CompletedProcess[str]) -> None:
    raise AssertionError(
        f"{name} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _encode_in_a_fresh_process(ckpt_dir: Path, wav: Path, out_path: Path) -> None:
    result = _run_worker(
        "_encode_determinism_worker",
        ["--ckpt-dir", str(ckpt_dir), "--wav", str(wav), "--out", str(out_path)],
    )
    if result.returncode != 0:
        _fail_worker("encode worker", result)


def test_same_reference_wav_encodes_identically_across_processes(
    breeze_model, tmp_path
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if REFERENCE_WAV is None or not REFERENCE_WAV.is_file():
        pytest.skip(
            f"bench reference wav not found (REFERENCE_WAV={REFERENCE_WAV}); "
            "set REFERENCE_VOICES_DIR to a directory containing eric/eric.wav"
        )

    # Each worker is its own fresh Python process with its own fresh cuDNN autotune
    # state -- exactly the "server restart" this bug is about. Not repeated calls in
    # one process, which would share whatever algorithm the first call picked.
    outputs = []
    for i in range(NUM_WORKERS):
        out_path = tmp_path / f"codes_{i}.pt"
        _encode_in_a_fresh_process(breeze_model, REFERENCE_WAV, out_path)
        outputs.append(torch.load(out_path))

    for i, result in enumerate(outputs):
        assert result["benchmark"] is False, (
            f"worker {i}'s codec encode ran with cudnn.benchmark=True -- "
            "encode_prompt_waveform's scoped flags did not take effect"
        )
        assert result["deterministic"] is True, (
            f"worker {i}'s codec encode ran with cudnn.deterministic=False -- "
            "encode_prompt_waveform's scoped flags did not take effect"
        )

    first_codes = outputs[0]["codes"]
    for i, result in enumerate(outputs[1:], start=1):
        assert result["codes"].shape == first_codes.shape
        assert torch.equal(result["codes"], first_codes), (
            f"worker {i} produced different codec codes than worker 0 for the same "
            "reference wav across fresh processes -- encode_prompt_waveform's scoped "
            "cudnn flags should make this process-independent"
        )


def test_multi_request_stream_runtime_sets_cudnn_benchmark_for_fast_codec(
    breeze_model,
) -> None:
    """Cheap, separate proof that the real `MultiRequestStreamRuntime` (not a
    reimplementation of its logic) sets `cudnn.benchmark=True` for a fast codec on
    CUDA -- the process-wide condition the test above reproduces directly rather than
    by building the runtime, because building it is what used to cost ~85s per
    worker. See `_runtime_benchmark_flag_worker.py` for what's real here and what's
    stubbed out (and why the stub doesn't affect what's being checked). Must run on
    GPU: the condition itself is gated on `device.type == "cuda"`.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    result = _run_worker("_runtime_benchmark_flag_worker", ["--ckpt-dir", str(breeze_model)])
    if result.returncode != 0:
        _fail_worker("runtime benchmark-flag worker", result)
