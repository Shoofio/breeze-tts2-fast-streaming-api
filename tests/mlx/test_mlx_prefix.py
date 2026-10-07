"""The MLX runtime's reference prefixes on the real checkpoint (contracts/runtime-seam.md,
`build_reference_prefix` and `iter_audio_chunks(prefix=...)`). Needs BREEZE_MLX_MODEL on Apple
Silicon."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.synthesis import (
    CodesRef,
    NoRef,
    PrefixRef,
    prepare_piece,
    voice_prefix_inputs,
)
from models.mlx_streaming import _CONDITIONAL, _NEGATIVE, _Generation, _to_mlx_prompt

pytestmark = pytest.mark.mlx

REF_TEXT = "The lighthouse keeper climbed the stairs at dusk."
TEXT = "Hello there, this is a short test."
INSTRUCTION = "A calm, warm voice."
# A cap well past the sentence's length, so the recorded run ends at EOS.
CAP = 200
SHORT = 24


@pytest.fixture(scope="module")
def reference_codes(mlx_runtime) -> torch.Tensor:
    """Reference codes from speech the runtime synthesizes itself, so no audio file is needed."""
    inputs = prepare_piece(mlx_runtime.tokenizer, mlx_runtime.model, NoRef(), REF_TEXT, DEFAULT_INSTRUCTION, 1.0)
    audio = np.concatenate([chunk.audio for chunk in mlx_runtime.iter_audio_chunks(inputs, seed=1)])
    (codes,) = mlx_runtime.audio_tokenizer.encode(audio, sr=mlx_runtime.sample_rate)["audio_codes"]
    return codes


def as_numpy(array) -> np.ndarray:
    import mlx.core as mx

    return np.array(array.astype(mx.float32))


class Comparison:
    """The teacher-forced comparison of research/proto/diag_teacher.py: the inline path's and
    the prefix path's sampler inputs at one point, and whether an argmax mismatch is a tie."""

    def __init__(self) -> None:
        self.points = 0
        self.max_error = 0.0
        self.mismatches: list[dict[str, float | int | str]] = []

    def add(self, step: int, point: str, inline, prefixed) -> tuple[int, int]:
        a, b = as_numpy(inline)[0], as_numpy(prefixed)[0]
        finite = np.isfinite(a)
        # The masked ids are -inf in both paths, and nothing else is non-finite.
        assert np.array_equal(finite, np.isfinite(b)), (step, point)
        error = float(np.abs(a[finite] - b[finite]).max())
        self.points += 1
        self.max_error = max(self.max_error, error)
        top_a, top_b = int(np.argmax(a)), int(np.argmax(b))
        if top_a != top_b:
            first, second = np.sort(a[finite])[-2:][::-1]
            margin = float(first - second)
            self.mismatches.append(
                {"step": step, "point": point, "margin": margin, "error": error, "tie": margin <= 2 * error}
            )
        return top_a, top_b


def generation(runtime, inputs, *, frames: int, prefix=None) -> _Generation:
    """A generation for `inputs` with the runtime's defaults: the conditional row, and under CFG
    the negative row too, as `iter_audio_chunks` builds them."""
    model = runtime._mlx_model
    rows = [_CONDITIONAL, _NEGATIVE] if "cfg_negative_prompt_ids" in inputs else [_CONDITIONAL]
    return _Generation(
        model,
        [_to_mlx_prompt(inputs, keys, model.config) for keys in rows],
        frames=frames,
        guidance=float(inputs.get("cfg_scale", 1.0)),
        backbone_sampling=runtime._backbone_sampling,
        depth_sampling=runtime._depth_sampling,
        repetition_penalty=runtime.config.repetition_penalty,
        seed=42,
        prefix=prefix,
    )


@pytest.mark.parametrize("cfg_scale", [1.0, 4.0])
def test_prefix_matches_inline_reference_teacher_forced(mlx_runtime, reference_codes, cfg_scale: float) -> None:
    import mlx.core as mx

    runtime = mlx_runtime
    tokenizer, model = runtime.tokenizer, runtime.model
    inline = prepare_piece(tokenizer, model, CodesRef(reference_codes, REF_TEXT), TEXT, INSTRUCTION, cfg_scale)
    prefix_inputs, prefix_len = voice_prefix_inputs(tokenizer, model, reference_codes, REF_TEXT)
    prefix = runtime.build_reference_prefix(prefix_inputs)
    suffix = prepare_piece(tokenizer, model, PrefixRef(prefix, REF_TEXT), TEXT, INSTRUCTION, cfg_scale)
    assert prefix.prefix_len == prefix_len
    # The two paths run the same tokens: the prefix then the suffix is the inline prompt.
    assert torch.equal(torch.cat([prefix_inputs["input_ids"], suffix["input_ids"]], dim=1), inline["input_ids"])

    # Record one inline request, its backbone greedy (top_k 1) so that EOS is the backbone's
    # argmax at the step it ends; the depth decoder samples with its defaults.
    observed: list[torch.Tensor] = []
    for _ in runtime.iter_audio_chunks(inline, seed=42, top_k=1, max_new_tokens=CAP, token_observer=observed.append):
        pass
    recorded = [frame.tolist() for frame in observed]
    assert 0 < len(recorded) < CAP, "the recorded run must end at EOS"

    # Feed those frames through both paths and compare every sampling point.
    eos = runtime._mlx_model.vocab_size
    paths = [
        generation(runtime, inline, frames=len(recorded)),
        generation(runtime, suffix, frames=len(recorded), prefix=prefix),
    ]
    comparison = Comparison()
    eos_steps: list[int | None] = [None, None]
    for step, frame in enumerate([*recorded, None]):
        logits = [path.backbone_logits() for path in paths]
        tops = comparison.add(step, "backbone", *logits)
        for index, top in enumerate(tops):
            if top == eos and eos_steps[index] is None:
                eos_steps[index] = step
        if frame is None:
            break
        # The replay is the recorded run: the token it took is the inline path's argmax, or tied
        # with it (top_k 1 keeps every token tied for the top logit and the draw picks among them;
        # argmax takes the lowest index).
        inline_logits = as_numpy(logits[0])[0]
        assert inline_logits[frame[0]] == inline_logits.max(), step
        steps = [path.depth_steps(mx.array(frame[:1])) for path in paths]
        last = len(frame) - 1
        for codebook in range(1, last + 1):
            comparison.add(step, f"codebook {codebook}", *(run.logits(codebook) for run in steps))
            if codebook < last:
                for run in steps:
                    run.feed(mx.array(frame[codebook : codebook + 1]), codebook)
        codes = mx.array(frame, dtype=mx.int32)
        for path in paths:
            path.advance(codes)

    not_ties = [m for m in comparison.mismatches if not m["tie"]]
    widest = max((float(m["margin"]) for m in comparison.mismatches), default=0.0)
    print(
        f"cfg_scale {cfg_scale}: {len(recorded)} frames, {comparison.points} points, "
        f"max logit error {comparison.max_error:.4g}, {len(comparison.mismatches)} argmax mismatches "
        f"({len(not_ties)} not ties, widest inline margin {widest:.4g}), "
        f"EOS at step {eos_steps[0]} inline / {eos_steps[1]} prefix"
    )
    assert not_ties == []
    assert eos_steps[0] == len(recorded)
    assert eos_steps[1] == eos_steps[0]


def test_two_requests_share_one_prefix_unchanged(mlx_runtime, reference_codes) -> None:
    runtime = mlx_runtime
    prefix_inputs, _ = voice_prefix_inputs(runtime.tokenizer, runtime.model, reference_codes, REF_TEXT)
    prefix = runtime.build_reference_prefix(prefix_inputs)
    config = runtime.model.config
    keys, _ = prefix.kv[0]
    assert len(prefix.kv) == config.num_hidden_layers
    assert tuple(keys.shape) == (1, config.num_key_value_heads, prefix.prefix_len, config.head_dim)
    before = [(as_numpy(keys), as_numpy(values)) for keys, values in prefix.kv]

    for cfg_scale in (1.0, 4.0):
        inputs = prepare_piece(
            runtime.tokenizer, runtime.model, PrefixRef(prefix, REF_TEXT), TEXT, INSTRUCTION, cfg_scale
        )
        chunks = list(runtime.iter_audio_chunks(inputs, prefix=prefix, seed=3, max_new_tokens=SHORT))
        assert chunks and chunks[-1].is_final
        assert np.concatenate([chunk.audio for chunk in chunks]).size > 0

    after = [(as_numpy(keys), as_numpy(values)) for keys, values in prefix.kv]
    assert all(
        np.array_equal(k0, k1) and np.array_equal(v0, v1) for (k0, v0), (k1, v1) in zip(before, after)
    )


def test_a_prefix_that_leaves_no_room_is_refused(mlx_runtime) -> None:
    codes = torch.zeros((2100, mlx_runtime.model.config.num_codebooks), dtype=torch.long)
    prefix_inputs, prefix_len = voice_prefix_inputs(mlx_runtime.tokenizer, mlx_runtime.model, codes, REF_TEXT)
    with pytest.raises(ValueError) as refused:
        mlx_runtime.build_reference_prefix(prefix_inputs)
    # fast_streaming.build_reference_prefix's message.
    assert str(refused.value) == (
        f"reference prefix of {prefix_len} tokens leaves no room to generate 12 frames after the "
        "shortest default-instruction suffix in the 2048-token context"
    )
