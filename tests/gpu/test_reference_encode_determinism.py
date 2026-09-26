"""GPU regression test for cross-process reference-encode determinism.

Finding (2026-09-25, user-approved deviation -- see
specs/003-cpp-compatible-api/research.md): with ``--fast-all``,
``torch.backends.cudnn.benchmark = True`` is set for the whole server process
(``models/stream_runtime/stream/runtime.py``'s ``MultiRequestStreamRuntime.__init__``).
cuDNN then autotunes whichever conv algorithm looks fastest *in that process* and
reuses it, and separate processes can autotune to different algorithms. Those
algorithms round differently, so about 1% of the fine-codebook codes (codebooks
6-15) come out different for the *same* reference wav across a server restart --
same wav, same text, same seed, different audio. ``breeze_infer/audio.py``'s
``encode_prompt_waveform`` now scopes ``benchmark=False, deterministic=True``
around just the codec's ``encode`` call to fix this. This test proves the fix
holds against the real fast codec runtime, not just the unit-level flag check in
``tests/test_audio.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_WAV = Path("$REFERENCE_VOICES_DIR/eric/eric.wav")


def _encode_in_a_fresh_process(ckpt_dir: Path, wav: Path, out_path: Path) -> None:
    # Run as `-m tests.gpu._encode_determinism_worker` with cwd=repo root (not a
    # bare script path) so the worker's `breeze_infer`/`models` imports resolve
    # the same way they do under `python -m pytest` from the repo root.
    result = subprocess.run(
        check=False,
        args=[
            sys.executable,
            "-m",
            "tests.gpu._encode_determinism_worker",
            "--ckpt-dir",
            str(ckpt_dir),
            "--wav",
            str(wav),
            "--out",
            str(out_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"encode worker failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def test_same_reference_wav_encodes_identically_across_process_restarts(
    breeze_model, tmp_path
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not REFERENCE_WAV.is_file():
        pytest.skip(f"bench reference wav not found at {REFERENCE_WAV}")

    first_out = tmp_path / "codes_a.pt"
    second_out = tmp_path / "codes_b.pt"

    # Two independent Python processes, each with its own fresh cuDNN autotune
    # state -- exactly the "server restart" this bug is about. Not two calls in
    # one process, which would share whatever algorithm the first call picked.
    _encode_in_a_fresh_process(breeze_model, REFERENCE_WAV, first_out)
    _encode_in_a_fresh_process(breeze_model, REFERENCE_WAV, second_out)

    codes_a = torch.load(first_out)
    codes_b = torch.load(second_out)

    assert codes_a.shape == codes_b.shape
    assert torch.equal(codes_a, codes_b), (
        "the same reference wav produced different codec codes across two fresh "
        "processes with the fast codec's cudnn.benchmark=True enabled -- "
        "encode_prompt_waveform's scoped cudnn flags should make this "
        "process-independent"
    )
