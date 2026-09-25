from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breeze_infer.templates import get_template
from models.cudagraph.depth_decoder_graph import DepthDecoderGraph
from models.cudagraph.sampling import (
    _UNSAMPLEABLE_TOKEN,
    MAX_REPETITION_PENALTY,
    MAX_TEMPERATURE,
    MIN_REPETITION_PENALTY,
    MIN_TEMPERATURE,
    NumberRule,
    _sample_logits_or_sentinel,
    apply_repetition_penalty,
    require_number,
    sample_logits,
)
from models.fast_streaming import (
    MIN_SUFFIX_FRAMES,
    MIN_SUFFIX_ROOM,
    FastBreezeStreamingRuntime,
    FastStreamingChunk,
    FastStreamingConfig,
    NonFiniteLogitsError,
    _branch_shape,
    _BranchBatch,
    _frame_flags,
    _get_dtype,
    is_backbone_eos_token,
    reject_dual_cfg,
    select_fast_cfg,
)


def test_runtime_dtype_follows_backbone_after_shared_lm_head_is_cast() -> None:
    class CompositeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.backbone_model = torch.nn.Linear(
                2, 2, bias=False, dtype=torch.bfloat16
            )

    model = CompositeModel()
    assert _get_dtype(model) == torch.bfloat16

    # BackboneGraph mutates the shared projection in exactly this way.
    model.lm_head.float()

    assert next(model.parameters()).dtype == torch.float32
    assert _get_dtype(model) == torch.bfloat16


def test_fast_streaming_defaults_to_repetition_penalty_1p1() -> None:
    assert FastStreamingConfig().repetition_penalty == 1.1


def test_fast_streaming_ceiling_defaults_to_the_server_maximum() -> None:
    # R12: 750 is the per-request default (from the model); 1500 the ceiling.
    assert FastStreamingConfig().max_new_tokens == 1500


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": float("nan")},
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"top_p": float("nan")},
        {"top_k": -1},
        {"top_k": 2.5},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("nan")},
        {"repetition_penalty": 1e-40},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_seq_len": 0},
    ],
)
def test_fast_streaming_config_rejects_invalid_fields(fields) -> None:
    with pytest.raises(ValueError, match=next(iter(fields))):
        FastStreamingConfig(**fields)


def test_fast_streaming_config_accepts_top_k_zero_as_disabled() -> None:
    assert FastStreamingConfig(top_k=0).top_k == 0


def _defaults_runtime(backbone: dict, depth: dict | None = None):
    runtime = object.__new__(FastBreezeStreamingRuntime)
    runtime.config = FastStreamingConfig()
    valid = {"temperature": 0.9, "top_k": 50, "top_p": 1.0, "do_sample": True}
    runtime.model = SimpleNamespace(
        generation_config=SimpleNamespace(**{**valid, "max_new_tokens": 750, **backbone}),
        depth_decoder=SimpleNamespace(
            generation_config=SimpleNamespace(**{**valid, **(depth or {})})
        ),
    )
    return runtime


def test_valid_model_generation_defaults_pass() -> None:
    _defaults_runtime({})._validate_model_defaults()
    # A model without its own length default falls back to the ceiling.
    _defaults_runtime({"max_new_tokens": None})._validate_model_defaults()


@pytest.mark.parametrize(
    ("backbone", "depth", "name"),
    [
        ({"temperature": float("nan")}, None, "temperature"),
        ({"top_p": float("nan")}, None, "top_p"),
        ({"top_k": -5}, None, "top_k"),
        ({"max_new_tokens": 0}, None, "max_new_tokens"),
        ({"max_new_tokens": 2.5}, None, "max_new_tokens"),
        ({}, {"temperature": float("nan")}, "temperature"),
        ({}, {"top_p": 0.0}, "top_p"),
    ],
)
def test_invalid_model_generation_defaults_are_rejected(backbone, depth, name) -> None:
    with pytest.raises(ValueError, match=name):
        _defaults_runtime(backbone, depth)._validate_model_defaults()


def test_fast_streaming_exposes_master_and_one_switch_per_stage() -> None:
    config = FastStreamingConfig()
    fast_fields = [
        name for name in config.__dataclass_fields__ if name.startswith("fast_")
    ]

    assert fast_fields == [
        "fast_all",
        "fast_text_encoder",
        "fast_backbone_prefill",
        "fast_backbone_decode",
        "fast_depth_decoder",
        "fast_codec",
    ]
    assert config.fast_all is None
    assert all(getattr(config, name) is False for name in fast_fields[1:])


def test_fast_streaming_master_switch_overrides_stage_switches() -> None:
    all_eager = FastStreamingConfig(fast_all=False)
    all_fast = FastStreamingConfig(fast_all=True, fast_codec=False)

    assert all_eager.stage_fast("text_encoder") is False
    assert all_eager.stage_fast("codec") is False
    assert all_fast.stage_fast("codec") is True
    assert FastStreamingConfig(fast_codec=False).stage_fast("codec") is False


def test_fast_cfg_selects_no_cfg_by_default() -> None:
    cfg = select_fast_cfg({})

    assert cfg.mode == "no_cfg"
    assert cfg.guidance_scale == 1.0
    assert cfg.use_negative_as_main is False


def test_fast_cfg_selects_single_cfg_when_negative_prompt_is_present() -> None:
    cfg = select_fast_cfg(
        {
            "cfg_scale": 2.5,
            "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )

    assert cfg.mode == "single_cfg"
    assert cfg.guidance_scale == 2.5
    assert cfg.use_negative_as_main is False


def test_fast_cfg_zero_uses_negative_as_main() -> None:
    cfg = select_fast_cfg(
        {
            "cfg_scale": 0.0,
            "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )

    assert cfg.mode == "no_cfg"
    assert cfg.guidance_scale == 1.0
    assert cfg.use_negative_as_main is True


@pytest.mark.parametrize("cfg_scale", [0.0, 0, 2.5, 100.0, np.float32(7.5)])
def test_fast_cfg_accepts_scales_from_0_to_100(cfg_scale) -> None:
    cfg = select_fast_cfg(
        {
            "cfg_scale": cfg_scale,
            "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )

    if cfg_scale != 0:
        assert cfg.guidance_scale == pytest.approx(float(cfg_scale))


@pytest.mark.parametrize(
    "cfg_scale",
    [float("nan"), float("inf"), -0.5, 100.5, 10**400, "4", True, None],
)
def test_fast_cfg_rejects_scales_outside_0_to_100(cfg_scale) -> None:
    # Checked before the scale reaches a graph's guidance buffer.
    with pytest.raises(ValueError, match="cfg_scale"):
        select_fast_cfg(
            {
                "cfg_scale": cfg_scale,
                "cfg_negative_prompt_ids": torch.ones(1, 2, dtype=torch.long),
            }
        )


def test_fast_streaming_rejects_dual_cfg_fields() -> None:
    with pytest.raises(ValueError, match="dual CFG"):
        reject_dual_cfg({"cfg_scale_ref": 1.0, "cfg_scale_ins": 2.0})


def test_backbone_eos_and_pad_frame_are_distinct() -> None:
    config = SimpleNamespace(vocab_size=2051, codebook_pad_token_id=2050)

    assert is_backbone_eos_token(torch.tensor(2051), config)
    assert not is_backbone_eos_token(torch.tensor(0), config)
    clean = torch.tensor([False])
    assert _frame_flags(torch.zeros(16, dtype=torch.long), clean, config) == (
        False,
        False,
    )

    pad_frame = torch.full((16,), 2050, dtype=torch.long)
    assert _frame_flags(pad_frame, clean, config) == (True, False)
    # The same host read carries the depth decoder's non-finite-logits flag.
    assert _frame_flags(pad_frame, torch.tensor([True]), config) == (True, True)


def test_ref_edit_tata_negative_branch_is_clone_without_instruction() -> None:
    template = get_template("ref_edit_tata")
    request = {
        "text": "target",
        "instruction": "speak softly",
        "ref_audio_codes": np.zeros((4, 16), dtype=np.int16),
        "ref_text": "reference",
        "speaker": "S0",
    }

    positive = template.build_segments(request)
    negative = template.build_negative_segments(request)

    assert "<ins_bos>speak softly<ins_eos>target" in positive[-1]["text"]
    assert negative[-1]["text"] == "[S0]target"
    assert "<ins_bos>" not in negative[-1]["text"]


def test_single_cfg_merges_cond_and_uncond_in_one_batched_text_path() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def _merge_input_ids_with_input_values(self, **kwargs):
            self.calls.append(kwargs)
            return {"inputs_embeds": kwargs["input_ids"].unsqueeze(-1).float()}

    runtime = object.__new__(FastBreezeStreamingRuntime)
    runtime.model = FakeModel()
    runtime._fast_text_encoder = True
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "text_ids_mask": torch.ones(1, 3, dtype=torch.bool),
        "text_ids_len": torch.tensor([3]),
        "input_values": None,
        "cfg_scale": 2.0,
        "cfg_negative_prompt_ids": torch.tensor([[4, 5]]),
        "cfg_negative_prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "cfg_negative_text_ids_mask": torch.ones(1, 2, dtype=torch.bool),
        "cfg_negative_text_ids_len": torch.tensor([2]),
    }

    branch = runtime._build_branch_batch(inputs, _branch_shape(inputs))

    assert branch.branch_batch_size == 2
    assert len(runtime.model.calls) == 1
    call = runtime.model.calls[0]
    assert call["input_ids"].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert call["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert call["text_ids_len"].tolist() == [3, 2]
    assert branch.inputs_embeds[..., 0].tolist() == [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]]

    runtime.model.calls.clear()
    runtime._fast_text_encoder = False
    eager_branch = runtime._build_branch_batch(inputs, _branch_shape(inputs))

    assert eager_branch.branch_batch_size == 2
    assert len(runtime.model.calls) == 2



class _RecordingDecodeRuntime:
    """A CPU-only runtime whose model-edge pieces are fakes.

    Everything that needs CUDA (graphs, codec, prefill) is replaced, so the
    real iter_audio_chunks loop runs and every sampling call is recorded.
    ``default_max_new_tokens`` is the model's generation-config default.
    """

    VOCAB = 100

    def __init__(
        self,
        monkeypatch,
        config: FastStreamingConfig,
        *,
        default_max_new_tokens: int | None = 750,
    ) -> None:
        self.sample_calls: list[dict] = []
        self.depth_calls: list[dict] = []
        self.codec_events: list[str] = []
        runtime = object.__new__(FastBreezeStreamingRuntime)
        runtime.config = config
        runtime.device = torch.device("cpu")
        runtime._fast_backbone_prefill = False
        runtime._backbone_prefill_graphs = {}
        runtime._reserved_codec_token_ids = ()
        runtime._codec_chunk_frames = 1
        runtime._fast_text_encoder = False
        runtime.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=self.VOCAB, codebook_pad_token_id=99),
            generation_config=SimpleNamespace(
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                max_new_tokens=default_max_new_tokens,
            ),
            depth_decoder=SimpleNamespace(
                generation_config=SimpleNamespace(
                    temperature=0.7, top_k=30, top_p=0.8, do_sample=True
                )
            ),
        )
        hidden = torch.zeros(1, 1, 4)
        logits = torch.zeros(1, self.VOCAB + 1)
        runtime._ensure_graphs = lambda *args, **kwargs: None
        runtime._codec = lambda: SimpleNamespace(
            open_request=lambda *args, **kwargs: self.codec_events.append("open"),
            close_request=lambda *args, **kwargs: self.codec_events.append("close"),
        )
        runtime._build_branch_batch = lambda inputs, shape: _BranchBatch(
            hidden, torch.ones(1, 1, dtype=torch.long), 1, select_fast_cfg(inputs)
        )
        runtime._run_prefill = lambda branch, prefix: (
            hidden,
            logits,
            branch.attention_mask,
            1,
            "eager",
        )
        runtime._backbone_graph = SimpleNamespace(
            set_generation_state=lambda mask: None,
            run=lambda frame, step_idx: (hidden, logits),
        )

        def depth_run(depth_hidden, token_batch, **kwargs):
            self.depth_calls.append(kwargs)
            return torch.zeros(1, 3, dtype=torch.long)

        runtime._depth_decoder_graph = SimpleNamespace(
            run=depth_run, nonfinite_logits=torch.zeros(1, dtype=torch.bool)
        )
        runtime._decode_codec_frames = lambda *, frames, is_final, timing, **kwargs: (
            FastStreamingChunk(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=24000,
                codec_frames=len(frames),
                is_final=is_final,
                timing=timing,
            )
        )

        # Tokens the fake sampler returns, in order, before falling back to 1.
        self.next_tokens: list[int] = []

        def recording_sample_logits(logits, **kwargs):
            self.sample_calls.append(kwargs)
            token = self.next_tokens.pop(0) if self.next_tokens else 1
            return torch.tensor([token])

        monkeypatch.setattr(
            "models.fast_streaming._sample_logits_or_sentinel", recording_sample_logits
        )
        self.runtime = runtime

    def run(self, **overrides) -> list[FastStreamingChunk]:
        # A one-token prompt, matching the fake prefill's cache length of 1.
        inputs = {"attention_mask": torch.ones(1, 1, dtype=torch.long)}
        return list(self.runtime.iter_audio_chunks(inputs, request_id="r", **overrides))


def _frames(chunks: list[FastStreamingChunk]) -> int:
    return sum(chunk.codec_frames for chunk in chunks)


def test_iter_audio_chunks_without_overrides_uses_the_model_defaults(
    monkeypatch,
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch,
        FastStreamingConfig(max_new_tokens=1500, repetition_penalty=1.1),
        default_max_new_tokens=5,
    )

    chunks = harness.run()

    assert _frames(chunks) == 5
    first, *decode = harness.sample_calls
    assert (first["temperature"], first["top_k"], first["top_p"]) == (0.9, 50, 0.95)
    assert all(call["repetition_penalty"] == 1.1 for call in decode)


def test_iter_audio_chunks_applies_overrides_to_backbone_only(monkeypatch) -> None:
    config = FastStreamingConfig(max_new_tokens=1500, repetition_penalty=1.1)
    harness = _RecordingDecodeRuntime(monkeypatch, config, default_max_new_tokens=5)
    generation_config = vars(harness.runtime.model.generation_config).copy()
    config_snapshot = dataclasses.asdict(config)

    chunks = harness.run(
        temperature=0.3, top_k=7, top_p=0.5, repetition_penalty=1.5, max_new_tokens=3
    )

    assert _frames(chunks) == 3
    assert chunks[-1].is_final
    for call in harness.sample_calls:
        assert (call["temperature"], call["top_k"], call["top_p"]) == (0.3, 7, 0.5)
    assert [call["repetition_penalty"] for call in harness.sample_calls[1:]] == [
        1.5,
        1.5,
    ]
    for call in harness.depth_calls:
        assert (call["temperature"], call["top_k"], call["top_p"]) == (0.7, 30, 0.8)
    # Overrides are per request: shared config is not mutated.
    assert vars(harness.runtime.model.generation_config) == generation_config
    assert dataclasses.asdict(harness.runtime.config) == config_snapshot
    harness.sample_calls.clear()
    harness.run()
    assert harness.sample_calls[0]["temperature"] == 0.9


def test_default_length_is_the_generation_config_value_not_the_ceiling(
    monkeypatch,
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=10), default_max_new_tokens=6
    )

    assert _frames(harness.run()) == 6
    # An explicit request may go past the default, up to the ceiling.
    assert _frames(harness.run(max_new_tokens=8)) == 8


def test_iter_audio_chunks_cannot_exceed_the_configured_ceiling(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=4), default_max_new_tokens=750
    )

    assert _frames(harness.run(max_new_tokens=50)) == 4
    # The model default is clamped too.
    assert _frames(harness.run()) == 4


def test_missing_generation_config_default_falls_back_to_the_ceiling(
    monkeypatch,
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=5), default_max_new_tokens=None
    )

    assert _frames(harness.run()) == 5


@pytest.mark.parametrize("temperature", [1e-40, 5e-324, 1e-6, MIN_TEMPERATURE])
def test_tiny_temperature_samples_finite_logits_greedily(temperature) -> None:
    logits = torch.tensor([[30.0, -30.0, 29.0, 0.0]])

    # 1e-40 itself overflows logits / temperature to inf and softmax to NaN.
    assert torch.isnan(torch.softmax(logits / 1e-40, dim=-1)).any()
    token = sample_logits(
        logits, temperature=temperature, top_k=0, top_p=1.0, do_sample=True
    )

    assert token.tolist() == [0]


def test_runtime_limits_match_the_http_contract() -> None:
    # contracts/http-api.md: temperature (0, 10], repetition_penalty [1e-4, 10].
    assert MAX_TEMPERATURE == 10
    assert MAX_REPETITION_PENALTY == 10
    assert MIN_REPETITION_PENALTY == 1e-4


def test_tiny_temperature_override_is_passed_through_not_rejected(
    monkeypatch,
) -> None:
    # sample_logits floors it; the runtime only rejects values <= 0.
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=3)
    )

    harness.run(temperature=1e-40)

    assert [call["temperature"] for call in harness.sample_calls] == [1e-40] * 3
    # The depth decoder keeps its own default.
    assert all(call["temperature"] == 0.7 for call in harness.depth_calls)


def test_override_bounds_are_inclusive(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=2)
    )

    assert (
        _frames(
            harness.run(
                temperature=MAX_TEMPERATURE, repetition_penalty=MAX_REPETITION_PENALTY
            )
        )
        == 2
    )
    assert _frames(harness.run(repetition_penalty=MIN_REPETITION_PENALTY)) == 2


def test_sampling_accepts_the_largest_temperature() -> None:
    token = sample_logits(
        torch.tensor([[1.0, 2.0]]),
        temperature=MAX_TEMPERATURE,
        top_k=0,
        top_p=1.0,
        do_sample=True,
    )

    assert token.tolist()[0] in (0, 1)


def test_smallest_repetition_penalty_keeps_logits_finite() -> None:
    logits = torch.tensor([[30.0, -30.0, 29.0, 0.0]])
    history = torch.tensor([0, 1])

    penalised = apply_repetition_penalty(
        logits.clone(), history, MIN_REPETITION_PENALTY
    )

    assert torch.isfinite(penalised).all()
    assert penalised[0].tolist() == pytest.approx([3e5, -3e-3, 29.0, 0.0])


@pytest.mark.parametrize(
    "penalty",
    [
        # 0 or a negative used to be floored into the strongest reward.
        0.0,
        -1.0,
        1e-40,
        MIN_REPETITION_PENALTY * (1 - 1e-9),
        MAX_REPETITION_PENALTY * (1 + 1e-9),
        float("nan"),
        float("inf"),
        10**400,
        True,
    ],
)
def test_repetition_penalty_outside_its_range_is_rejected(penalty) -> None:
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="repetition_penalty"):
        apply_repetition_penalty(logits, torch.tensor([0]), penalty)
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_logits(
            logits,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            token_history=torch.tensor([0]),
            repetition_penalty=penalty,
        )


def test_repetition_penalty_none_is_disabled() -> None:
    logits = torch.tensor([[1.0, -2.0]])

    assert torch.equal(
        apply_repetition_penalty(logits.clone(), torch.tensor([0, 1]), None), logits
    )


@pytest.mark.parametrize(
    "temperature",
    [
        float("nan"),
        float("inf"),
        0.0,
        -1.0,
        10**400,
        1e39,
        MAX_TEMPERATURE * (1 + 1e-9),
    ],
)
def test_invalid_sampling_temperature_is_rejected(temperature) -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_logits(
            torch.tensor([[1.0, 2.0]]),
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )


_UNSAMPLEABLE_ROWS = [
    [1.0, float("nan"), 2.0],
    [1.0, float("inf"), 2.0],
    # Every token masked: softmax of all -inf is NaN.
    [float("-inf"), float("-inf"), float("-inf")],
]


def test_nonfinite_logits_error_is_a_server_error_not_a_bad_request() -> None:
    # A route maps ValueError to 400; a model producing NaN is not the
    # client's fault.
    assert issubclass(NonFiniteLogitsError, RuntimeError)
    assert not issubclass(NonFiniteLogitsError, ValueError)


@pytest.mark.parametrize("do_sample", [True, False])
@pytest.mark.parametrize("bad_row", _UNSAMPLEABLE_ROWS)
def test_unsampleable_logits_give_the_sentinel_token_without_a_host_sync(
    monkeypatch, bad_row, do_sample
) -> None:
    # multinomial would raise on CPU, and on CUDA a device-side assert would
    # poison the whole process; argmax would pick an arbitrary token. The
    # runtime's sampler marks the row on the device instead, and the loop
    # checks it at the host read it already makes (the EOS check).
    def no_host_read(*args, **kwargs):
        raise AssertionError("the sampler must not read a tensor on the host")

    for name in ("__bool__", "item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, no_host_read)
    logits = torch.tensor([bad_row, [0.0, 5.0, 0.0]])

    token = _sample_logits_or_sentinel(
        logits, temperature=1.0, top_k=0, top_p=1.0, do_sample=do_sample
    )

    monkeypatch.undo()
    assert token.tolist()[0] == _UNSAMPLEABLE_TOKEN
    # The other row is sampled normally.
    assert 0 <= token.tolist()[1] < 3


@pytest.mark.parametrize("do_sample", [True, False])
@pytest.mark.parametrize("bad_row", _UNSAMPLEABLE_ROWS)
def test_public_sample_logits_raises_on_unsampleable_logits(bad_row, do_sample) -> None:
    with pytest.raises(NonFiniteLogitsError):
        sample_logits(
            torch.tensor([bad_row]),
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=do_sample,
        )


def test_overflow_from_the_temperature_floor_is_caught() -> None:
    # Finite logits, but 3e34 / 1e-5 = 3e39 overflows float32 to inf.
    logits = torch.tensor([[3e34, 1.0, 0.0, 1.0]])

    token = _sample_logits_or_sentinel(
        logits, temperature=MIN_TEMPERATURE, top_k=0, top_p=1.0, do_sample=True
    )

    assert token.tolist() == [_UNSAMPLEABLE_TOKEN]
    with pytest.raises(NonFiniteLogitsError):
        sample_logits(
            logits, temperature=MIN_TEMPERATURE, top_k=0, top_p=1.0, do_sample=True
        )


def test_overflow_from_the_repetition_penalty_is_caught() -> None:
    # 3.4e34 / 1e-4 = 3.4e38 is finite, but 3.4e35 / 1e-4 overflows.
    logits = torch.tensor([[3.4e35, 1.0, 0.0]])

    token = _sample_logits_or_sentinel(
        logits,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        do_sample=False,
        token_history=torch.tensor([0]),
        repetition_penalty=MIN_REPETITION_PENALTY,
    )

    assert token.tolist() == [_UNSAMPLEABLE_TOKEN]


@pytest.mark.parametrize("do_sample", [True, False])
def test_masked_tokens_alone_do_not_trip_the_logits_guard(do_sample) -> None:
    token = sample_logits(
        torch.tensor([[float("-inf"), 1.0, float("-inf")]]),
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        do_sample=do_sample,
    )

    assert token.tolist() == [1]


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        (1.0, True),
        (np.float32(0.5), True),
        (0.0, False),
        (float("nan"), False),
        (10**400, False),
        (True, False),
        ("1", False),
    ],
)
def test_require_number_is_the_one_shared_check(value, valid) -> None:
    rule = NumberRule(integer=False, minimum=0.0, minimum_inclusive=False)
    if valid:
        require_number("x", value, rule)
    else:
        with pytest.raises(ValueError, match="x must be"):
            require_number("x", value, rule)


def _depth_sampler(temperature: float, *, do_sample: bool = True):
    """A DepthDecoderGraph with only the state its captured _cfg_sample reads."""
    graph = object.__new__(DepthDecoderGraph)
    graph.nonfinite_logits = torch.zeros(1, dtype=torch.bool)
    graph.batch_size = 1
    graph.half = 1
    graph.codec_codebook_size = 4
    graph.vocab_size = 4
    graph.guidance_scale = torch.ones(1, 1)
    graph.temperature_buf = torch.full((1, 1), temperature)
    graph.top_k_buf = torch.zeros(1, 1, dtype=torch.long)
    graph._max_k = 4
    graph._topk_ranks = torch.arange(4)
    graph.top_p_buf = torch.ones(1, 1)
    graph.do_sample_buf = torch.full((1,), int(do_sample), dtype=torch.long)
    graph._debug_probs_slot = None
    graph._tok_buf = torch.zeros(1, dtype=torch.long)
    return graph


@pytest.mark.parametrize("temperature", [1e-40, 5e-324])
def test_depth_decoder_sampling_floors_a_tiny_temperature(temperature) -> None:
    # FastStreamingConfig(temperature=1e-40) is valid (> 0) and reaches the
    # depth decoder's temperature buffer too.
    assert FastStreamingConfig(temperature=temperature).temperature == temperature
    graph = _depth_sampler(temperature)

    graph._cfg_sample(torch.tensor([[[30.0, -30.0, 29.0, 0.0]]]))

    assert graph._tok_buf.tolist() == [0]
    assert graph.nonfinite_logits.tolist() == [False]


@pytest.mark.parametrize("do_sample", [True, False])
@pytest.mark.parametrize(
    "row",
    [
        [float("nan")] * 4,
        [float("inf"), float("nan"), float("-inf"), float("nan")],
        [float("-inf")] * 4,
        # Any NaN or +inf in a row fails it, even beside finite entries.
        [float("nan"), 30.0, 1.0, 0.0],
        [float("inf"), 30.0, 1.0, 0.0],
    ],
)
def test_depth_decoder_flags_a_row_with_a_non_finite_logit(row, do_sample) -> None:
    graph = _depth_sampler(0.9, do_sample=do_sample)
    graph._tok_buf.fill_(3)

    graph._cfg_sample(torch.tensor([[row]]))

    # A fixed, valid code instead of a device assert, and a flag the runtime
    # reads at its per-frame host sync.
    assert graph._tok_buf.tolist() == [0]
    assert graph.nonfinite_logits.tolist() == [True]
    # The flag stays set for the rest of the frame's codebooks.
    graph._cfg_sample(torch.tensor([[[0.0, 5.0, 0.0, 0.0]]]))
    assert graph.nonfinite_logits.tolist() == [True]


@pytest.mark.parametrize("do_sample", [True, False])
def test_depth_decoder_allows_masked_entries(do_sample) -> None:
    graph = _depth_sampler(0.9, do_sample=do_sample)

    graph._cfg_sample(torch.tensor([[[float("-inf"), 30.0, float("-inf"), 0.0]]]))

    assert graph._tok_buf.tolist() == [1]
    assert graph.nonfinite_logits.tolist() == [False]


def test_depth_decoder_catches_overflow_from_the_temperature_floor() -> None:
    graph = _depth_sampler(MIN_TEMPERATURE)

    graph._cfg_sample(torch.tensor([[[3e34, 1.0, 0.0, 1.0]]]))

    assert graph.nonfinite_logits.tolist() == [True]
    assert graph._tok_buf.tolist() == [0]


@pytest.mark.parametrize(
    "row",
    [
        # Overflows only at the 1e-5 temperature floor, which greedy never uses.
        [3e34, 1.0, 0.0, 1.0],
        [float("-inf"), 30.0, float("-inf"), 0.0],
    ],
)
def test_depth_decoder_judges_greedy_rows_on_the_unscaled_logits(row) -> None:
    graph = _depth_sampler(MIN_TEMPERATURE, do_sample=False)

    graph._cfg_sample(torch.tensor([[row]]))

    assert graph.nonfinite_logits.tolist() == [False]
    assert graph._tok_buf.tolist() == [int(torch.tensor(row).argmax())]


@pytest.mark.parametrize(
    "row",
    [
        [float("nan"), 30.0, 1.0, 0.0],
        [float("inf"), 30.0, 1.0, 0.0],
        [float("-inf")] * 4,
    ],
)
def test_depth_decoder_still_flags_bad_greedy_rows(row) -> None:
    graph = _depth_sampler(MIN_TEMPERATURE, do_sample=False)

    graph._cfg_sample(torch.tensor([[row]]))

    assert graph.nonfinite_logits.tolist() == [True]
    assert graph._tok_buf.tolist() == [0]


def test_depth_decoder_eager_run_ignores_flags_from_a_rebuild_capture() -> None:
    # A batch size with no captured bucket rebuilds and recaptures the graph;
    # its warm-up passes run on dummy buffers and OR into the same flag. Only
    # the request's own pass may decide it.
    graph = object.__new__(DepthDecoderGraph)
    graph.nonfinite_logits = torch.zeros(1, dtype=torch.bool)
    graph.num_decode_codebooks = 3
    graph.device = "cpu"
    graph.bucket_sizes = [1, 2]
    graph.no_graph = False
    graph.half = 2
    graph.backbone_hidden_buf = torch.zeros(4, 4)
    graph.first_cb_token_buf = torch.zeros(4, dtype=torch.long)
    graph.output_tokens = torch.zeros(4, 3, dtype=torch.long)
    graph.static_cache = SimpleNamespace(reset=lambda: None)
    passes = []

    def rebuild_with_dirty_capture(batch_size):
        graph.nonfinite_logits.fill_(True)

    def real_pass():
        passes.append(graph.nonfinite_logits.tolist())

    graph.ensure_batch_size = rebuild_with_dirty_capture
    graph._full_loop = real_pass

    graph.run(torch.zeros(4, 4), torch.zeros(4, dtype=torch.long))

    # The request's own pass starts from a clear flag.
    assert passes == [[False]]
    assert graph.nonfinite_logits.tolist() == [False]


def test_depth_decoder_run_clears_a_stale_flag_even_on_its_early_return() -> None:
    graph = object.__new__(DepthDecoderGraph)
    graph.nonfinite_logits = torch.ones(1, dtype=torch.bool)
    graph.num_decode_codebooks = 3
    graph.device = "cpu"

    # An odd CFG batch takes run()'s early return, before any graph replays.
    graph.run(torch.zeros(3, 4), torch.zeros(3, dtype=torch.long))

    assert graph.nonfinite_logits.tolist() == [False]


def test_numpy_scalar_overrides_are_accepted(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=1500)
    )

    chunks = harness.run(
        temperature=np.float32(0.5),
        top_k=np.int64(7),
        top_p=np.float64(0.9),
        repetition_penalty=np.float32(1.25),
        max_new_tokens=np.int64(3),
    )

    assert _frames(chunks) == 3
    call = harness.sample_calls[-1]
    assert (call["temperature"], call["top_k"], call["top_p"]) == (0.5, 7, 0.9)
    assert type(call["top_k"]) is int
    assert call["repetition_penalty"] == 1.25
    assert _room_runtime().max_new_tokens_room(np.int64(5), _prompt(10)) == 5


def test_branch_shape_is_computed_once_per_request(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=2)
    )
    calls = []

    def counting(inputs):
        calls.append(inputs)
        return _branch_shape(inputs)

    monkeypatch.setattr("models.fast_streaming._branch_shape", counting)
    harness.run()

    assert len(calls) == 1


_INVALID_OVERRIDES = [
    {"temperature": 0.0},
    {"top_k": 0},
    {"top_p": -0.1},
    {"repetition_penalty": 0.0},
    {"max_new_tokens": 0},
    {"max_new_tokens": -3},
    {"temperature": float("nan")},
    {"temperature": float("inf")},
    {"top_p": float("nan")},
    {"top_p": float("inf")},
    {"repetition_penalty": float("nan")},
    {"repetition_penalty": float("inf")},
    {"top_k": float("nan")},
    {"max_new_tokens": float("inf")},
    # Counts must be integers; the runtime does not truncate them.
    {"top_k": 2.5},
    {"max_new_tokens": 2.5},
    # Absurd magnitudes are a caller error, not a numerics problem.
    {"temperature": 1e39},
    {"temperature": MAX_TEMPERATURE * (1 + 1e-9)},
    {"repetition_penalty": MAX_REPETITION_PENALTY * (1 + 1e-9)},
    {"temperature": 1e4},
    {"repetition_penalty": 1e4},
    # Non-numbers and bools are rejected as ValueError, never TypeError.
    {"temperature": "0.5"},
    {"top_p": "x"},
    {"repetition_penalty": [1.1]},
    {"top_k": "5"},
    {"max_new_tokens": "5"},
    {"temperature": True},
    {"top_p": True},
    {"repetition_penalty": False},
    {"top_k": True},
    {"max_new_tokens": True},
    {"temperature": np.bool_(True)},
    {"top_k": np.bool_(True)},
    # Below the floor a penalty overflows the logits it divides.
    {"repetition_penalty": 1e-40},
    {"repetition_penalty": MIN_REPETITION_PENALTY * (1 - 1e-9)},
    # Too large for a float: rejected, not an OverflowError.
    {"temperature": 10**400},
    {"top_p": 10**400},
    {"repetition_penalty": -(10**400)},
]


@pytest.mark.parametrize("override", _INVALID_OVERRIDES)
def test_iter_audio_chunks_rejects_invalid_overrides(monkeypatch, override) -> None:
    harness = _RecordingDecodeRuntime(monkeypatch, FastStreamingConfig())

    with pytest.raises(ValueError, match=next(iter(override))):
        harness.run(**override)
    assert harness.sample_calls == []
    assert harness.codec_events == []


@pytest.mark.parametrize("bad_step", [0, 2])
def test_unsampleable_backbone_token_raises_at_the_eos_check(
    monkeypatch, bad_step
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=10)
    )
    # Token 0 comes from the prefill; token n from decode step n - 1.
    harness.next_tokens = [1] * bad_step + [_UNSAMPLEABLE_TOKEN]
    chunks = []

    with pytest.raises(NonFiniteLogitsError, match="cannot sample"):
        # extend keeps the chunks yielded before the error.
        chunks.extend(
            harness.runtime.iter_audio_chunks(
                {"attention_mask": torch.ones(1, 1, dtype=torch.long)}, request_id="r"
            )
        )

    # The sentinel never reaches the depth decoder or the codec.
    assert len(harness.depth_calls) == bad_step
    assert _frames(chunks) == bad_step
    assert harness.codec_events == ["open", "close"]


def test_depth_decoder_nonfinite_flag_raises_before_the_frame_is_used(
    monkeypatch,
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=10)
    )
    depth_graph = harness.runtime._depth_decoder_graph
    depth_run = depth_graph.run

    def flagging_run(*args, **kwargs):
        tokens = depth_run(*args, **kwargs)
        if len(harness.depth_calls) == 2:
            depth_graph.nonfinite_logits.fill_(True)
        return tokens

    depth_graph.run = flagging_run
    observed = []
    chunks = []

    with pytest.raises(NonFiniteLogitsError, match="depth decoder"):
        chunks.extend(
            harness.runtime.iter_audio_chunks(
                {"attention_mask": torch.ones(1, 1, dtype=torch.long)},
                request_id="r",
                token_observer=observed.append,
            )
        )

    # The flagged frame is neither observed nor decoded.
    assert len(observed) == 1
    assert _frames(chunks) == 1
    assert harness.codec_events == ["open", "close"]


def test_setup_failure_before_decoding_does_not_leak_a_codec_request(
    monkeypatch,
) -> None:
    harness = _RecordingDecodeRuntime(monkeypatch, FastStreamingConfig())
    # The depth decoder's sampling parameters are read during request setup.
    harness.runtime.model.depth_decoder.generation_config.temperature = "broken"

    with pytest.raises(ValueError):
        harness.run()
    assert harness.codec_events.count("open") == harness.codec_events.count("close")


def test_prefill_failure_closes_the_codec_request(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(monkeypatch, FastStreamingConfig())

    def failing_prefill(branch, prefix):
        raise RuntimeError("CUDA error")

    harness.runtime._run_prefill = failing_prefill

    with pytest.raises(RuntimeError, match="CUDA error"):
        harness.run()
    assert harness.codec_events == ["open", "close"]


def test_prompt_that_leaves_no_room_raises_before_streaming(monkeypatch) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=500, max_seq_len=128)
    )
    runtime = harness.runtime
    hidden = torch.zeros(1, 1, 4)
    logits = torch.zeros(1, harness.VOCAB + 1)
    runtime._build_branch_batch = lambda inputs, shape: _BranchBatch(
        hidden, inputs["attention_mask"], 1, select_fast_cfg(inputs)
    )
    runtime._run_prefill = lambda branch, prefix: (
        hidden,
        logits,
        branch.attention_mask,
        int(branch.attention_mask.shape[1]),
        "eager",
    )

    # 127 tokens fill the cache to max_seq_len - 1: the room is 0.
    assert runtime.max_new_tokens_room(None, _prompt(127)) == 0
    with pytest.raises(ValueError, match="no room"):
        list(runtime.iter_audio_chunks(_prompt(127), request_id="r"))
    assert harness.codec_events == []
    assert harness.sample_calls == []
    # One token shorter leaves exactly one frame.
    assert _frames(list(runtime.iter_audio_chunks(_prompt(126), request_id="r"))) == 1


def _room_runtime(
    *,
    max_seq_len: int = 2048,
    max_new_tokens: int = 1500,
    default_max_new_tokens: int = 750,
    fast_backbone_prefill: bool = False,
    prefill_graphs: dict | None = None,
) -> FastBreezeStreamingRuntime:
    runtime = object.__new__(FastBreezeStreamingRuntime)
    runtime.config = FastStreamingConfig(
        max_seq_len=max_seq_len, max_new_tokens=max_new_tokens
    )
    runtime.model = SimpleNamespace(
        generation_config=SimpleNamespace(max_new_tokens=default_max_new_tokens)
    )
    runtime._fast_backbone_prefill = fast_backbone_prefill
    runtime._backbone_prefill_graphs = prefill_graphs or {}
    return runtime


def _prompt(length: int, negative_length: int | None = None) -> dict:
    inputs = {"attention_mask": torch.ones(1, length, dtype=torch.long)}
    if negative_length is not None:
        inputs.update(
            cfg_scale=2.0,
            cfg_negative_prompt_ids=torch.ones(1, negative_length, dtype=torch.long),
            cfg_negative_prompt_attention_mask=torch.ones(
                1, negative_length, dtype=torch.long
            ),
        )
    return inputs


def test_room_without_a_request_uses_the_generation_config_default() -> None:
    assert _room_runtime().max_new_tokens_room(None, _prompt(100)) == 750
    assert (
        _room_runtime(default_max_new_tokens=300).max_new_tokens_room(
            None, _prompt(100)
        )
        == 300
    )


def test_room_accepts_a_huge_integer_request_and_clamps_it() -> None:
    assert _room_runtime().max_new_tokens_room(10**400, _prompt(100)) == 1500


@pytest.mark.parametrize(
    "requested", [0, -3, float("nan"), float("inf"), 2.5, "5", True, np.bool_(True)]
)
def test_room_rejects_an_invalid_request(requested) -> None:
    # Mapping the wire's 0 to "use the default" is the boundary's job.
    with pytest.raises(ValueError, match="max_new_tokens"):
        _room_runtime().max_new_tokens_room(requested, _prompt(100))


@pytest.mark.parametrize(
    ("inputs", "shape"),
    [
        (_prompt(7), ("no_cfg", 1, 7)),
        # CFG left-pads both branches to the longer one.
        (_prompt(7, 9), ("single_cfg", 2, 9)),
        (_prompt(9, 7), ("single_cfg", 2, 9)),
        # cfg_scale 0 runs the negative prompt alone.
        ({**_prompt(7, 9), "cfg_scale": 0.0}, ("no_cfg", 1, 9)),
    ],
)
def test_branch_shape(inputs, shape) -> None:
    cfg, branch_batch_size, seq_len = _branch_shape(inputs)

    assert (cfg.mode, branch_batch_size, seq_len) == shape


def test_room_clamps_request_to_ceiling_and_context() -> None:
    runtime = _room_runtime()

    assert runtime.max_new_tokens_room(5000, _prompt(100)) == 1500
    assert runtime.max_new_tokens_room(1200, _prompt(100)) == 1200
    # The decode loop stops at prefill_len + step >= max_seq_len - 1.
    assert runtime.max_new_tokens_room(1500, _prompt(1000)) == 1047
    assert runtime.max_new_tokens_room(None, _prompt(2047)) <= 0


def test_room_clamps_the_default_to_the_ceiling() -> None:
    runtime = _room_runtime(max_new_tokens=500, default_max_new_tokens=900)

    assert runtime.max_new_tokens_room(None, _prompt(10)) == 500


def test_room_counts_the_longer_cfg_branch_and_the_prefix() -> None:
    runtime = _room_runtime()

    assert runtime.max_new_tokens_room(1500, _prompt(900, 1000)) == 1047
    assert runtime.max_new_tokens_room(1500, _prompt(1000, 900)) == 1047
    assert runtime.max_new_tokens_room(1500, _prompt(500), prefix_len=500) == 1047


def test_room_counts_graph_bucket_padding() -> None:
    # 1990 raw tokens would leave 57 frames, but the graph prefill pads the
    # prompt to its 2016-token bucket, so the loop really stops after 31.
    runtime = _room_runtime(fast_backbone_prefill=True)

    assert runtime.max_new_tokens_room(1500, _prompt(1990)) == 2048 - 2016 - 1
    assert runtime.max_new_tokens_room(1500, _prompt(1000), prefix_len=33) == (
        2048 - 33 - 1024 - 1
    )


def test_room_uses_the_exact_length_when_a_frozen_cache_falls_back_to_eager() -> None:
    class FrozenCache:
        frozen = True
        token_granularity = 32

        def __init__(self, fits: bool) -> None:
            self.fits = fits

        def has_bucket(self, batch_size, seq_len, prefix_len=0):
            return self.fits

    eager = _room_runtime(
        fast_backbone_prefill=True, prefill_graphs={1: FrozenCache(False)}
    )
    graph = _room_runtime(
        fast_backbone_prefill=True, prefill_graphs={1: FrozenCache(True)}
    )

    assert eager.max_new_tokens_room(1500, _prompt(1990)) == 2048 - 1990 - 1
    assert graph.max_new_tokens_room(1500, _prompt(1990)) == 2048 - 2016 - 1


class _FakePrefillCache:
    """A prefill graph cache with a fixed set of captured ``(batch, bucket)`` keys."""

    token_granularity = 32

    def __init__(
        self, buckets, *, frozen: bool = True, max_seq_len: int = 2048
    ) -> None:
        self.buckets = set(buckets)
        self.frozen = frozen
        self.max_seq_len = max_seq_len
        self.calls: list[int] = []

    def _bucket(self, seq_len: int) -> int:
        return -(-seq_len // self.token_granularity) * self.token_granularity

    def has_bucket(self, batch_size, seq_len, prefix_len=0) -> bool:
        bucket = self._bucket(seq_len)
        return (batch_size, bucket) in self.buckets and (
            prefix_len + bucket <= self.max_seq_len
        )

    def __call__(self, inputs_embeds, attention_mask, *, prefix_len=0):
        self.calls.append(int(attention_mask.shape[1]))
        return SimpleNamespace(
            prefill_len=prefix_len + self._bucket(int(attention_mask.shape[1]))
        )


# The fast profile's frozen batch-1 prefill buckets (configs/fast.json).
_PROFILE_BUCKETS = {(1, n) for n in range(32, 513, 32)}


@pytest.mark.parametrize(
    ("fast", "cache", "seq_len", "prefix_len", "plan"),
    [
        # Fast prefill off: always eager at the exact length.
        (False, None, 40, 0, (False, 40)),
        # No cache yet: _run_prefill creates an unfrozen one and pads.
        (True, None, 40, 0, (True, 64)),
        (True, None, 40, 100, (True, 164)),
        # Frozen profile cache: graph when a bucket fits, else exact eager.
        (True, _FakePrefillCache(_PROFILE_BUCKETS), 500, 0, (True, 512)),
        (True, _FakePrefillCache(_PROFILE_BUCKETS), 600, 0, (False, 600)),
        # Unfrozen cache: a bucket past max_seq_len runs eagerly, not raises.
        (True, _FakePrefillCache(set(), frozen=False), 1000, 1030, (False, 2030)),
        (True, None, 1000, 1030, (False, 2030)),
        (True, _FakePrefillCache(set(), frozen=False), 1000, 900, (True, 1924)),
        # A bucket that would leave fewer than MIN_SUFFIX_FRAMES frames runs
        # eagerly at the exact length instead, which leaves more.
        (True, _FakePrefillCache(set(), frozen=False), 1000, 1024, (False, 2024)),
        (True, None, 10, 2004, (False, 2014)),
        (True, None, 10, 2003, (True, 2035)),
    ],
)
def test_prefill_plan(fast, cache, seq_len, prefix_len, plan) -> None:
    runtime = _room_runtime(
        fast_backbone_prefill=fast, prefill_graphs=None if cache is None else {1: cache}
    )

    assert runtime._prefill_plan(1, seq_len, prefix_len) == plan


@pytest.mark.parametrize(
    ("requested", "prompt_len", "prefix_len", "fast_prefill"),
    [
        (None, 10, 0, False),
        # Graph bucket padding: 40 tokens fill 64 slots.
        (None, 40, 0, True),
        (None, 20, 12, True),
        (None, 90, 0, True),
        # The request's own cap binds before the context does.
        (5, 10, 0, True),
    ],
)
def test_room_is_exactly_the_frames_the_decode_loop_produces(
    monkeypatch, requested, prompt_len, prefix_len, fast_prefill
) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=500, max_seq_len=128)
    )
    runtime = harness.runtime
    runtime._fast_backbone_prefill = fast_prefill
    runtime._backbone_prefill_graphs = {}
    hidden = torch.zeros(1, 1, 4)
    logits = torch.zeros(1, harness.VOCAB + 1)
    runtime._build_branch_batch = lambda inputs, shape: _BranchBatch(
        hidden, inputs["attention_mask"], 1, select_fast_cfg(inputs)
    )

    # The real _run_prefill's cache length is _prefill_plan's; only the model
    # work is faked. The prefix is passed by length since KV loading is faked.
    def run_prefill(branch, prefix):
        use_graph, prefill_len = runtime._prefill_plan(
            1, int(branch.attention_mask.shape[1]), prefix_len
        )
        path = "graph" if use_graph else "eager"
        return hidden, logits, branch.attention_mask, prefill_len, path

    runtime._run_prefill = run_prefill
    inputs = _prompt(prompt_len)

    room = runtime.max_new_tokens_room(requested, inputs, prefix_len=prefix_len)
    chunks = list(
        runtime.iter_audio_chunks(inputs, request_id="r", max_new_tokens=room)
    )

    assert room > 0
    assert _frames(chunks) == room
    if requested is None:
        # Uncapped, the loop's own context check stops at the same frame.
        uncapped = list(runtime.iter_audio_chunks(inputs, request_id="r"))
        assert _frames(uncapped) == room


def _prefix_runtime(
    prefix_len: int, prefill_cache: _FakePrefillCache | None = None
) -> FastBreezeStreamingRuntime:
    """A CPU runtime whose reference-prefix paths return fake KV.

    With ``prefill_cache`` the fast backbone prefill is on and uses it.
    """
    runtime = object.__new__(FastBreezeStreamingRuntime)
    runtime.config = FastStreamingConfig(max_seq_len=2048, max_new_tokens=1500)
    runtime.dtype = torch.float32
    runtime._fast_backbone_prefill = prefill_cache is not None
    runtime._backbone_prefill_graphs = (
        {} if prefill_cache is None else {1: prefill_cache}
    )
    runtime._ensure_graphs = lambda *args, **kwargs: None
    graph_kv = torch.ones(1, 2, 2048, 3)
    runtime._backbone_graph = SimpleNamespace(
        num_layers=1,
        static_cache=SimpleNamespace(
            layers=[SimpleNamespace(keys=graph_kv, values=graph_kv)]
        ),
    )
    runtime._merge_branch = lambda **kwargs: (
        torch.zeros(1, prefix_len, 4),
        torch.ones(1, prefix_len, dtype=torch.long),
    )
    kv = torch.zeros(1, 2, prefix_len, 3)
    runtime.eager_calls = 0

    def backbone_model(**kwargs):
        runtime.eager_calls += 1
        return SimpleNamespace(past_key_values=[(kv, kv)])

    runtime.model = SimpleNamespace(backbone_model=backbone_model)
    return runtime


# _merge_branch is faked, so the prefix inputs only need their keys.
_PREFIX_INPUTS = dict.fromkeys(
    ("input_ids", "attention_mask", "text_ids_mask", "text_ids_len")
)


# The longest prefix that still leaves room for the default-instruction suffix
# and MIN_SUFFIX_FRAMES frames: the decode loop needs
# prefix + suffix + frames <= max_seq_len - 1.
_LONGEST_PREFIX = 2048 - 1 - MIN_SUFFIX_ROOM


def test_min_suffix_room_leaves_about_a_second_of_audio() -> None:
    # 12 frames at the codec's 12.5 Hz is about 1 s.
    assert MIN_SUFFIX_FRAMES == 12
    assert MIN_SUFFIX_ROOM > MIN_SUFFIX_FRAMES


# The default-instruction suffix MIN_SUFFIX_ROOM is built from.
_MIN_SUFFIX_TOKENS = MIN_SUFFIX_ROOM - MIN_SUFFIX_FRAMES


@pytest.mark.parametrize("prefix_len", range(2004, _LONGEST_PREFIX + 1))
def test_every_accepted_prefix_keeps_its_min_suffix_frames(prefix_len) -> None:
    # 2004-2017 used to take the graph path: the 10-token suffix padded to 32
    # left 2048 - prefix - 33 < 12 frames.
    runtime = _room_runtime(fast_backbone_prefill=True)

    room = runtime.max_new_tokens_room(
        None, _prompt(_MIN_SUFFIX_TOKENS), prefix_len=prefix_len
    )

    assert room >= MIN_SUFFIX_FRAMES


@pytest.mark.parametrize("prefix_len", [0, 1000, 1990, 2004])
def test_room_never_grows_with_a_longer_prompt(prefix_len) -> None:
    runtime = _room_runtime(fast_backbone_prefill=True)
    rooms = [
        runtime.max_new_tokens_room(1500, _prompt(length), prefix_len=prefix_len)
        for length in range(1, 2048 - prefix_len)
    ]

    assert all(later <= earlier for earlier, later in itertools.pairwise(rooms))


def test_reference_prefix_is_not_limited_by_the_max_new_tokens_ceiling() -> None:
    # 600 + the 1500 ceiling exceeds 2048, but the prefix still leaves room.
    prefix = _prefix_runtime(600).build_reference_prefix(_PREFIX_INPUTS)

    assert prefix.prefix_len == 600
    longest = _prefix_runtime(_LONGEST_PREFIX).build_reference_prefix(_PREFIX_INPUTS)
    assert longest.prefix_len == _LONGEST_PREFIX


@pytest.mark.parametrize("prefix_len", [_LONGEST_PREFIX + 1, 2040, 2046, 2047])
def test_reference_prefix_without_room_for_a_minimal_suffix_is_rejected(
    prefix_len,
) -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        _prefix_runtime(prefix_len).build_reference_prefix(_PREFIX_INPUTS)


def test_reference_prefix_longer_than_every_frozen_bucket_runs_eagerly() -> None:
    # 600 tokens pad to a 608 bucket the fast profile never captured; replaying
    # it would raise "backbone prefill CUDA graph (1, 608) was not declared".
    cache = _FakePrefillCache(_PROFILE_BUCKETS)
    runtime = _prefix_runtime(600, cache)

    prefix = runtime.build_reference_prefix(_PREFIX_INPUTS)

    assert cache.calls == []
    assert runtime.eager_calls == 1
    assert prefix.prefix_len == 600
    assert prefix.kv.shape == (1, 2, 2, 600, 3)


def test_reference_prefix_with_a_frozen_bucket_replays_the_graph() -> None:
    cache = _FakePrefillCache(_PROFILE_BUCKETS)
    runtime = _prefix_runtime(500, cache)

    prefix = runtime.build_reference_prefix(_PREFIX_INPUTS)

    assert cache.calls == [500]
    assert runtime.eager_calls == 0
    # The KV comes from the graph's static cache (ones), not the eager path.
    assert prefix.kv.shape == (1, 2, 2, 500, 3)
    assert bool((prefix.kv == 1).all())
