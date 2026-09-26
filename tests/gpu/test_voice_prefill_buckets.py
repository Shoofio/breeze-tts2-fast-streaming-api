"""A saved-voice suffix longer than 256 tokens must still replay a CUDA graph, ported from
`api-alignment:tests/gpu/test_voice_prefill_buckets.py` (tasks.md T067).

Before review the continuation buckets stopped at 256 while full prefill covered 512, so long
instruction+text requests silently fell back to eager. The prefill buckets are shared, so
every declared bucket serves suffixes too; that is still true on this branch's runtime
(`models/fast_streaming.py`), just behind a different low-level shape:
`FastBreezeStreamingRuntime._build_branch_batch` now takes an explicit `_BranchShape` (the
module-level `_branch_shape(inputs)`, not derived inside the method), and
`templates.prepare_prefix_inputs`/`prepare_suffix_inputs` no longer take an `audio_tokenizer`
argument. Those are the only two changes from the old test; the graph-bucket assertions
themselves (`_prefill_plan`, `BackbonePrefillGraphCache.has_bucket`/`.replays`, `_run_prefill`'s
`path`) are unchanged.

The voice is registered through `POST /v1/voices` (this port's `_voice_components`/
`register_voice`, shared with `test_voice_equivalence.py`), and its resolved codes/ref_text
are read back off the live registry (`components.voices.get().registry.lookup`) for the
direct runtime calls below -- the same in-process access `tests/gpu/test_speech_long_text.py`
uses to spy on `routes_speech.prepare_piece`. A final `POST /v1/audio/speech` with the same
long text confirms the whole route -- not just the runtime call this test drives directly --
produces plausible audio through the shared bucket.
"""

from __future__ import annotations

import pytest
import torch

from breeze_infer.runtime import set_all_seeds
from breeze_infer.templates import prepare_prefix_inputs, prepare_suffix_inputs
from models.fast_streaming import _branch_shape
from tests.gpu.test_voice_equivalence import (  # noqa: F401
    register_voice,
    speak,
    voices_app,
)

pytestmark = pytest.mark.gpu

SENTENCE = (
    "The committee reviewed the proposal carefully, weighed every objection, "
    "and finally agreed to publish the revised schedule next week. "
)
INSTRUCTION = "Read this as a measured, unhurried announcement."


def test_warmup_manifest_has_one_prefill_graph_set(gpu_env) -> None:
    prefill = gpu_env.manifest["stages"]["backbone_prefill"]

    assert "continuation_graphs" not in prefill
    declared = {(g["branch_batch_size"], g["sequence_length"]) for g in prefill["graphs"]}
    assert declared == {(b, n) for b in (1, 2) for n in range(32, 513, 32)}


@torch.inference_mode()
def test_long_suffix_replays_the_shared_prefill_graph(
    gpu_env, voices_app, reference_clips  # noqa: F811 (voices_app is the imported fixture)
) -> None:
    client, events, components = voices_app
    env = gpu_env
    runtime = env.runtime
    clip = reference_clips[0]

    record = register_voice(client, clip.audio, clip.sample_rate, clip.ref_text, name="bucket-voice")
    resolved = components.voices.get().registry.lookup(record["id"])
    assert resolved is not None

    prefix = runtime.build_reference_prefix(
        prepare_prefix_inputs(
            env.tokenizer,
            env.model,
            {"id": "prefix", "speaker": "S0", "ref_text": resolved.ref_text, "ref_audio_codes": resolved.codes},
        )
    )

    request = {
        "id": "long-suffix",
        "text": (SENTENCE * 14).strip(),
        "instruction": INSTRUCTION,
        "speaker": "S0",
    }
    inputs = prepare_suffix_inputs(env.tokenizer, env.model, request, guidance_scale=1.0)
    suffix_len = int(inputs["attention_mask"].shape[1])
    assert 256 < suffix_len <= 512, suffix_len

    runtime._ensure_graphs(1, 1.0)
    cache = runtime._backbone_prefill_graph
    assert cache is not None and cache.frozen
    assert cache.has_bucket(1, suffix_len, prefix.prefix_len)
    replays_before = cache.replays

    shape = _branch_shape(inputs)
    branch = runtime._build_branch_batch(inputs, shape)
    _, _, _, prefill_len, path = runtime._run_prefill(branch, prefix)

    assert path == "graph"
    assert cache.replays == replays_before + 1
    assert prefill_len == prefix.prefix_len + cache._bucket(suffix_len)

    set_all_seeds(7)
    iterator = runtime.iter_audio_chunks(
        inputs, request_id="gpu-long-suffix", seed=7, token_observer=None, prefix=prefix
    )
    try:
        first = next(iterator)
    finally:
        iterator.close()
    assert first.timing["prefill_path"] == "graph"
    assert first.timing["prefill_gpu_ms"] > 0

    # The same request through the real route, with the registered voice_id rather than the
    # bare codes used above: production never calls _build_branch_batch/_run_prefill itself,
    # so this is what actually proves the shared bucket serves a real request end to end.
    samples, accepted, body = speak(
        client, events, voice_id=record["id"], text=request["text"], instruction=INSTRUCTION,
        seed=7, max_new_tokens=100,
    )
    assert accepted["reference"] == "voice_prefix", accepted
    assert len(body) > 0 and len(body) % 2 == 0
    assert samples.any(), "audio is entirely silence"
