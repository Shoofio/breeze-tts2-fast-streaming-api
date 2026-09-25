"""Per-request overrides and the eager fallbacks on the real model.

The fast profile freezes its text-encoder and backbone-prefill graphs at 512
tokens. A single piece longer than that used to raise ``RuntimeError: ... has
no fitting bucket``, and a longer reference prefix ``... was not declared in
the warmup profile``; both must now run eagerly and still produce audio.
"""

from __future__ import annotations

import pytest
import torch

from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.runtime import set_all_seeds
from breeze_infer.templates import (
    get_template,
    prepare_inputs,
    prepare_prefix_inputs,
    prepare_suffix_inputs,
)
from models.cudagraph.sampling import (
    _UNSAMPLEABLE_TOKEN,
    MAX_REPETITION_PENALTY,
    MAX_TEMPERATURE,
    MIN_REPETITION_PENALTY,
    _sample_logits_or_sentinel,
    sample_logits,
)
from models.fast_streaming import (
    MIN_SUFFIX_FRAMES,
    MIN_SUFFIX_ROOM,
    NonFiniteLogitsError,
)

pytestmark = pytest.mark.gpu

SENTENCE = (
    "The committee reviewed the proposal carefully, weighed every objection, "
    "and finally agreed to publish the revised schedule next week. "
)


def _inputs(env, text: str):
    return prepare_inputs(
        env.tokenizer,
        env.model,
        [
            {
                "id": "overrides",
                "text": text,
                "instruction": "Speak clearly and naturally.",
                "speaker": "S0",
            }
        ],
        get_template("tts_instruction"),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )


def _run(env, inputs, *, seed: int = 7, **overrides):
    set_all_seeds(seed)
    frames = []
    chunks = list(
        env.runtime.iter_audio_chunks(
            inputs,
            request_id="gpu-overrides",
            seed=seed,
            token_observer=frames.append,
            **overrides,
        )
    )
    return chunks, frames


def test_overlong_piece_falls_back_to_eager_text_encoder(gpu_env) -> None:
    text = (SENTENCE * 25).strip()
    token_count = len(gpu_env.tokenizer(text)["input_ids"])
    assert token_count > 512, "the piece must exceed every warmed text-encoder bucket"

    # A frozen text-encoder cache counts a miss each time it has no bucket and
    # hands the segment to the eager path.
    text_cache = gpu_env.model._fast_text_encoder_graph_cache
    misses_before = text_cache.misses

    chunks, frames = _run(gpu_env, _inputs(gpu_env, text), max_new_tokens=3)

    assert text_cache.misses > misses_before
    assert 0 < len(frames) <= 3
    assert sum(chunk.audio.size for chunk in chunks) > 0


def _backbone_tokens(frames) -> list[int]:
    return [int(frame[0]) for frame in frames]


def test_greedy_override_makes_the_first_backbone_token_seed_independent(
    gpu_env,
) -> None:
    inputs = _inputs(gpu_env, "The cat sat on the mat.")
    first_tokens = set()
    for seed in (7, 8, 9):
        _, frames = _run(gpu_env, inputs, seed=seed, top_k=1, max_new_tokens=2)
        first_tokens.add(int(frames[0][0]))

    # top_k=1 leaves one candidate, so the RNG cannot change the prefill token.
    assert len(first_tokens) == 1


def test_sampling_overrides_change_the_backbone_trace_for_the_same_seed(
    gpu_env,
) -> None:
    inputs = _inputs(gpu_env, "The cat sat on the mat.")
    _, default_frames = _run(gpu_env, inputs, max_new_tokens=4)
    _, flat_frames = _run(
        gpu_env,
        inputs,
        # The flattest distribution the contract allows, over every codebook
        # entry: with the same seed, matching the default trace would be a fluke.
        temperature=MAX_TEMPERATURE,
        top_k=2048,
        top_p=1.0,
        repetition_penalty=1.3,
        max_new_tokens=4,
    )

    assert 0 < len(flat_frames) <= 4
    assert _backbone_tokens(flat_frames) != _backbone_tokens(default_frames)


def test_tiny_temperature_is_floored_instead_of_producing_nan(gpu_env) -> None:
    # 1e-40 passes validation (finite, > 0) but logits / 1e-40 overflow to inf
    # and softmax to NaN, which used to crash torch.multinomial mid-stream.
    inputs = _inputs(gpu_env, "The cat sat on the mat.")

    chunks, frames = _run(gpu_env, inputs, temperature=1e-40, max_new_tokens=4)

    assert 0 < len(frames) <= 4
    assert sum(chunk.audio.size for chunk in chunks) > 0


@torch.inference_mode()
def test_reference_prefix_longer_than_every_prefill_bucket_builds_eagerly(
    gpu_env, reference_clips
) -> None:
    env = gpu_env
    runtime = env.runtime
    clip = reference_clips[0]
    reference = {
        "id": "long-prefix",
        "speaker": "S0",
        # Long enough to pass every warmed bucket (512) and the old guard,
        # which rejected any prefix over max_seq_len - 1500 = 548 tokens.
        "ref_text": (SENTENCE * 30).strip(),
        "ref_audio_codes": clip.codes,
    }
    runtime._ensure_graphs(1, 1.0)
    cache = runtime._backbone_prefill_graph
    assert cache is not None and cache.frozen
    replays_before = cache.replays

    prefix = runtime.build_reference_prefix(
        prepare_prefix_inputs(env.tokenizer, env.model, reference)
    )

    assert prefix.prefix_len > 548, "the prefix must exceed the old guard"
    assert cache.replays == replays_before
    inputs = prepare_suffix_inputs(
        env.tokenizer,
        env.model,
        {
            **reference,
            "text": "The cat sat on the mat.",
            "instruction": "Speak clearly.",
        },
        guidance_scale=1.0,
    )
    room = runtime.max_new_tokens_room(4, inputs, prefix_len=prefix.prefix_len)
    assert room == 4
    set_all_seeds(7)
    frames = []
    chunks = list(
        runtime.iter_audio_chunks(
            inputs,
            request_id="gpu-long-prefix",
            seed=7,
            token_observer=frames.append,
            prefix=prefix,
            max_new_tokens=room,
        )
    )

    # The short suffix still replays a graph after the eager-built prefix.
    assert chunks[0].timing["prefill_path"] == "graph"
    assert 0 < len(frames) <= 4
    assert sum(chunk.audio.size for chunk in chunks) > 0


def test_largest_allowed_temperature_and_penalty_still_sample(gpu_env) -> None:
    # The contract's ceilings (10); the logits flatten but stay finite, so
    # every step samples a valid token.
    inputs = _inputs(gpu_env, "The cat sat on the mat.")

    chunks, frames = _run(
        gpu_env,
        inputs,
        temperature=MAX_TEMPERATURE,
        repetition_penalty=MAX_REPETITION_PENALTY,
        max_new_tokens=4,
    )

    assert 0 < len(frames) <= 4
    assert sum(chunk.audio.size for chunk in chunks) > 0


@torch.inference_mode()
def test_captured_depth_decoder_flags_nan_logits_instead_of_asserting(
    gpu_env,
) -> None:
    # A NaN backbone hidden state makes every depth logit NaN. Before the
    # guard, multinomial on it was a device-side assert that poisoned the
    # process; now the graph samples fixed codes and raises its flag.
    graph = gpu_env.runtime._depth_decoder_graph
    hidden_size = graph.backbone_hidden_size
    hidden = torch.full(
        (1, hidden_size), float("nan"), dtype=graph.dtype, device=graph.device
    )
    first = torch.zeros(1, dtype=torch.long, device=graph.device)

    tokens = graph.run(hidden, first)

    assert bool(graph.nonfinite_logits.item())
    assert tokens.cpu().tolist() == [[0] * graph.num_decode_codebooks]
    # The next run clears the flag, and the CUDA context is still usable.
    graph.run(torch.zeros_like(hidden), first)
    assert not bool(graph.nonfinite_logits.item())
    _, frames = _run(gpu_env, _inputs(gpu_env, "Hello."), max_new_tokens=2)
    assert frames


@pytest.mark.parametrize("fill", [float("nan"), 3e34])
def test_backbone_sampler_catches_nan_and_overflow_on_cuda(gpu_env, fill) -> None:
    # 3e34 is finite but overflows at the 1e-5 temperature floor.
    logits = torch.full((1, 2049), fill, device="cuda")
    sampling = {"temperature": 1e-5, "top_k": 50, "top_p": 1.0, "do_sample": True}

    token = _sample_logits_or_sentinel(logits, **sampling)

    assert token.cpu().tolist() == [_UNSAMPLEABLE_TOKEN]
    with pytest.raises(NonFiniteLogitsError):
        sample_logits(logits, **sampling)


def test_smallest_allowed_repetition_penalty_still_samples(gpu_env) -> None:
    # MIN_REPETITION_PENALTY multiplies the real positive logits of repeated
    # tokens by 1e4; they must stay finite for the whole stream.
    inputs = _inputs(gpu_env, "The cat sat on the mat.")

    chunks, frames = _run(
        gpu_env, inputs, repetition_penalty=MIN_REPETITION_PENALTY, max_new_tokens=8
    )

    assert 0 < len(frames) <= 8
    assert sum(chunk.audio.size for chunk in chunks) > 0


def test_min_suffix_room_is_the_default_instruction_suffix_plus_min_frames(
    gpu_env,
) -> None:
    # The suffix a registered voice must leave room for: a one-word text with
    # the default instruction on the guided branch, plus about 1 s of frames.
    env = gpu_env
    for text in ("a", "Hi", "嗨"):
        request = {
            "id": "min-suffix",
            "speaker": "S0",
            "ref_text": "x",
            "text": text,
            "instruction": DEFAULT_INSTRUCTION,
        }
        inputs = prepare_suffix_inputs(
            env.tokenizer, env.model, request, guidance_scale=1.0
        )
        guided = int(inputs["attention_mask"].shape[1])

        assert MIN_SUFFIX_ROOM == guided + MIN_SUFFIX_FRAMES, text
