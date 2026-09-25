"""Any ``cfg_scale`` runs on the graphs warmed for the profile's scales (R12 point 6).

The warmup profile only exercises 1.0 and 4.0, but the guidance scale is a
runtime tensor in the captured backbone graph and a plain argument to the depth
decoder, so other values must neither fail nor capture anything new. 0 selects
the negative prompt as the only branch (batch 1); the others run both branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu.conftest import synthesize

pytestmark = pytest.mark.gpu

REQUEST = {
    "id": "cfg-values",
    "text": "The cat sat on the mat.",
    "instruction": "Speak clearly and naturally.",
    "speaker": "S0",
}


def _graph_state(env) -> dict:
    """Every captured graph the runtime holds, by key and object identity."""
    runtime = env.runtime
    return {
        "backbone": {size: id(graph) for size, graph in runtime._backbone_graphs.items()},
        "backbone_prefill": {
            size: cache.graph_keys
            for size, cache in runtime._backbone_prefill_graphs.items()
        },
        "depth_decoder": sorted(runtime._depth_decoder_graph._bucket_graphs),
        "text_encoder": env.model._fast_text_encoder_graph_cache.graph_keys,
    }


@pytest.mark.parametrize("cfg_scale", [2.5, 7.5, 0.0])
def test_cfg_scale_produces_finite_audio_without_recapture(gpu_env, cfg_scale) -> None:
    before = _graph_state(gpu_env)

    audio, frames = synthesize(
        gpu_env, dict(REQUEST), cfg=cfg_scale, seed=11, max_frames=12
    )

    assert frames, "no codec frame was generated"
    assert audio.size > 0
    assert np.isfinite(audio).all()
    assert _graph_state(gpu_env) == before
    if cfg_scale != 0.0:
        # The two-branch backbone graph now carries this request's scale.
        guidance = gpu_env.runtime._backbone_graphs[2].guidance_scale
        assert float(guidance.item()) == pytest.approx(cfg_scale)
