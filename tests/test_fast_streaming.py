from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breeze_infer.templates import get_template
from models.cudagraph.sampling import sample_logits
from models.fast_streaming import (
    _MIN_BACKBONE_TEMPERATURE,
    FastBreezeStreamingRuntime,
    FastStreamingChunk,
    FastStreamingConfig,
    _BranchBatch,
    _get_dtype,
    is_backbone_eos_token,
    is_terminal_pad_frame,
    reject_dual_cfg,
    select_fast_cfg,
    should_decode_codec_frame,
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


def test_fast_streaming_rejects_dual_cfg_fields() -> None:
    with pytest.raises(ValueError, match="dual CFG"):
        reject_dual_cfg({"cfg_scale_ref": 1.0, "cfg_scale_ins": 2.0})


def test_backbone_eos_and_pad_frame_are_distinct() -> None:
    config = SimpleNamespace(vocab_size=2051, codebook_pad_token_id=2050)

    assert is_backbone_eos_token(torch.tensor(2051), config)
    assert not is_backbone_eos_token(torch.tensor(0), config)
    assert not is_terminal_pad_frame(torch.zeros(16, dtype=torch.long), config)

    pad_frame = torch.full((16,), 2050, dtype=torch.long)
    assert is_terminal_pad_frame(pad_frame, config)
    assert not should_decode_codec_frame(pad_frame, config)


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

    branch = runtime._build_branch_batch(inputs)

    assert branch.branch_batch_size == 2
    assert len(runtime.model.calls) == 1
    call = runtime.model.calls[0]
    assert call["input_ids"].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert call["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert call["text_ids_len"].tolist() == [3, 2]
    assert branch.inputs_embeds[..., 0].tolist() == [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]]

    runtime.model.calls.clear()
    runtime._fast_text_encoder = False
    eager_branch = runtime._build_branch_batch(inputs)

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
        runtime = object.__new__(FastBreezeStreamingRuntime)
        runtime.config = config
        runtime.device = torch.device("cpu")
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
            open_request=lambda *args, **kwargs: None,
            close_request=lambda *args, **kwargs: None,
        )
        runtime._build_branch_batch = lambda inputs: _BranchBatch(
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

        runtime._depth_decoder_graph = SimpleNamespace(run=depth_run)
        runtime._decode_codec_frames = lambda *, frames, is_final, timing, **kwargs: (
            FastStreamingChunk(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=24000,
                codec_frames=len(frames),
                is_final=is_final,
                timing=timing,
            )
        )

        def recording_sample_logits(logits, **kwargs):
            self.sample_calls.append(kwargs)
            return torch.tensor([1])

        monkeypatch.setattr(
            "models.fast_streaming.sample_logits", recording_sample_logits
        )
        self.runtime = runtime

    def run(self, **overrides) -> list[FastStreamingChunk]:
        return list(self.runtime.iter_audio_chunks({}, request_id="r", **overrides))


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


@pytest.mark.parametrize("temperature", [1e-40, 5e-324, 1e-6])
def test_tiny_backbone_temperature_is_floored(monkeypatch, temperature) -> None:
    harness = _RecordingDecodeRuntime(
        monkeypatch, FastStreamingConfig(max_new_tokens=3)
    )

    harness.run(temperature=temperature)

    assert [call["temperature"] for call in harness.sample_calls] == [
        _MIN_BACKBONE_TEMPERATURE
    ] * 3
    # The depth decoder keeps its own default.
    assert all(call["temperature"] == 0.7 for call in harness.depth_calls)


def test_temperature_floor_samples_finite_logits_greedily() -> None:
    logits = torch.tensor([[30.0, -30.0, 29.0, 0.0]])

    # 1e-40 itself overflows logits / temperature to inf and softmax to NaN.
    assert torch.isnan(torch.softmax(logits / 1e-40, dim=-1)).any()
    token = sample_logits(
        logits,
        temperature=_MIN_BACKBONE_TEMPERATURE,
        top_k=0,
        top_p=1.0,
        do_sample=True,
    )

    assert token.tolist() == [0]


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
]


@pytest.mark.parametrize("override", _INVALID_OVERRIDES)
def test_iter_audio_chunks_rejects_invalid_overrides(monkeypatch, override) -> None:
    harness = _RecordingDecodeRuntime(monkeypatch, FastStreamingConfig())

    with pytest.raises(ValueError, match=next(iter(override))):
        harness.run(**override)
    assert harness.sample_calls == []


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


@pytest.mark.parametrize("requested", [0, -3, float("nan"), float("inf"), 2.5])
def test_room_rejects_an_invalid_request(requested) -> None:
    # Mapping the wire's 0 to "use the default" is the boundary's job.
    with pytest.raises(ValueError, match="max_new_tokens"):
        _room_runtime().max_new_tokens_room(requested, _prompt(100))


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
        (True, _FakePrefillCache(set(), frozen=False), 1000, 1024, (True, 2048)),
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
    runtime._build_branch_batch = lambda inputs: _BranchBatch(
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


def test_reference_prefix_is_not_limited_by_the_max_new_tokens_ceiling() -> None:
    # 600 + the 1500 ceiling exceeds 2048, but the prefix still leaves room.
    prefix = _prefix_runtime(600).build_reference_prefix(_PREFIX_INPUTS)

    assert prefix.prefix_len == 600
    assert _prefix_runtime(2046).build_reference_prefix(_PREFIX_INPUTS).prefix_len == 2046


def test_reference_prefix_that_leaves_no_slot_is_rejected() -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        _prefix_runtime(2047).build_reference_prefix(_PREFIX_INPUTS)


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
