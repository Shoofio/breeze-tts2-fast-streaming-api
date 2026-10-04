"""The MLX runtime's room rules (research R4): CUDA's exact-length rule, with no mlx and no model.

`MlxBreezeStreamingRuntime` is built here with no loaded parts: the room methods read only the
runtime's `FastStreamingConfig` and the checkpoint's `generation_config.json` default, so this
runs on every platform.
"""

from __future__ import annotations

import pytest

from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from models.fast_streaming import PromptLength
from models.mlx_streaming import MlxBreezeStreamingRuntime
from tests.fakes import FakeRuntime, FakeStreamingConfig

MAX_SEQ_LEN = 2048
DEFAULT_MAX_NEW_TOKENS = 750


def room_runtime() -> MlxBreezeStreamingRuntime:
    # The checkpoint's generation_config.json is the only input the room rules read.
    return MlxBreezeStreamingRuntime(
        mlx_model=None,
        codec=None,
        tokenizer=None,
        model_config=None,
        generation_config={"max_new_tokens": DEFAULT_MAX_NEW_TOKENS},
    )


def test_frame_cap_defaults_to_the_checkpoint_value() -> None:
    assert room_runtime().frame_cap(None) == DEFAULT_MAX_NEW_TOKENS


def test_frame_cap_clamps_to_the_ceiling() -> None:
    assert room_runtime().frame_cap(5000) == MAX_NEW_TOKENS_CEILING


def test_short_prompt_is_capped_by_the_frame_cap() -> None:
    room = room_runtime().room_for_length(None, PromptLength(1, 100), prefix_len=0)
    assert room == min(DEFAULT_MAX_NEW_TOKENS, MAX_SEQ_LEN - 100 - 1)


def test_prefix_counts_towards_the_context() -> None:
    room = room_runtime().room_for_length(None, PromptLength(1, 500), prefix_len=1500)
    assert room == MAX_SEQ_LEN - 2000 - 1 == 47


@pytest.mark.parametrize(("prefix_len", "seq_len"), [(1547, 500), (1600, 500), (0, 2047), (0, 3000)])
def test_no_room_once_the_prompt_fills_the_context(prefix_len: int, seq_len: int) -> None:
    room = room_runtime().room_for_length(None, PromptLength(1, seq_len), prefix_len=prefix_len)
    assert room <= 0


@pytest.mark.parametrize("requested", [0, -1, 1.5, True, "10"])
def test_invalid_request_is_refused_as_on_cuda(requested: object) -> None:
    with pytest.raises(ValueError):
        room_runtime().room_for_length(requested, PromptLength(1, 100))  # type: ignore[arg-type]


@pytest.mark.parametrize("requested", [None, 1, 24, 750, 1499, 1500, 5000])
@pytest.mark.parametrize("branch_batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 31, 100, 500, 1024, 2000, 2047, 2100])
@pytest.mark.parametrize("prefix_len", [0, 300, 1500, 2015])
def test_room_matches_the_fake_runtimes_exact_length_model(
    requested: int | None, branch_batch_size: int, seq_len: int, prefix_len: int
) -> None:
    fake = FakeRuntime(
        config=FakeStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS_CEILING,
            max_seq_len=MAX_SEQ_LEN,
            fast_backbone_prefill=False,
        ),
        default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )
    length = PromptLength(branch_batch_size, seq_len)
    assert room_runtime().room_for_length(
        requested, length, prefix_len=prefix_len
    ) == fake.room_for_length(requested, length, prefix_len=prefix_len)
