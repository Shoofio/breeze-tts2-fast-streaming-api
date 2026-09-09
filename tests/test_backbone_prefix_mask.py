from __future__ import annotations

import torch

from models.cudagraph.backbone_graph import decode_valid_keys
from models.cudagraph.backbone_prefill_graph import (
    continuation_allowed_keys,
    continuation_positions,
)


def _valid(pad_lens, prefix_lens, position, max_seq_len=16):
    return decode_valid_keys(
        torch.arange(max_seq_len),
        torch.tensor(pad_lens),
        torch.tensor(prefix_lens),
        position,
    )


def test_decode_mask_without_prefix_matches_the_contiguous_rule() -> None:
    valid = _valid(pad_lens=[3, 0], prefix_lens=[0, 0], position=9)

    expected_row0 = [(3 <= k <= 9) for k in range(16)]
    expected_row1 = [(0 <= k <= 9) for k in range(16)]
    assert valid[0].tolist() == expected_row0
    assert valid[1].tolist() == expected_row1


def test_decode_mask_with_prefix_admits_prefix_and_masks_the_hole() -> None:
    # prefix [0, 5), bucket of 6 at [5, 11): row 0 has 4 real tokens, row 1 has 2.
    prefix, bucket = 5, 6
    real = [4, 2]
    pad_lens = [prefix + bucket - r for r in real]
    position = prefix + bucket  # first decode slot

    valid = _valid(pad_lens=pad_lens, prefix_lens=[prefix, prefix], position=position)

    for row, r in enumerate(real):
        hole = range(prefix, prefix + bucket - r)
        expected = [
            (k < prefix) or (prefix + bucket - r <= k <= position) for k in range(16)
        ]
        assert valid[row].tolist() == expected
        assert not any(valid[row, k] for k in hole)
        assert valid[row, :prefix].all()
        assert valid[row, position]
        assert not valid[row, position + 1 :].any()


def test_decode_mask_accepts_a_tensor_position() -> None:
    valid = _valid(pad_lens=[0], prefix_lens=[0], position=torch.tensor(4))

    assert valid[0].tolist() == [k <= 4 for k in range(16)]


def test_continuation_mask_opens_prefix_and_keeps_causal_order() -> None:
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    prefix_len, key_len = 3, 12

    allowed = continuation_allowed_keys(attention_mask, key_len, prefix_len)

    assert tuple(allowed.shape) == (2, 4, key_len)
    # every query sees the whole prefix
    assert allowed[:, :, :prefix_len].all()
    # nothing beyond the bucket
    assert not allowed[:, :, prefix_len + 4 :].any()
    # row 0: padding at bucket positions 0 and 1 is never a key
    assert not allowed[0, :, prefix_len + 0].any()
    assert not allowed[0, :, prefix_len + 1].any()
    # causal within the bucket for the real tokens
    assert allowed[0, 2, prefix_len + 2] and not allowed[0, 2, prefix_len + 3]
    assert allowed[0, 3, prefix_len + 2] and allowed[0, 3, prefix_len + 3]
    assert allowed[1, 1, prefix_len + 0] and allowed[1, 1, prefix_len + 1]
    assert not allowed[1, 1, prefix_len + 2]


def test_continuation_mask_with_zero_prefix_is_the_full_prefill_mask() -> None:
    attention_mask = torch.tensor([[0, 1, 1]])

    allowed = continuation_allowed_keys(attention_mask, 8, 0)

    expected = torch.zeros(3, 8, dtype=torch.bool)
    expected[1, 1] = True
    expected[2, 1] = True
    expected[2, 2] = True
    assert torch.equal(allowed[0], expected)


def test_continuation_positions_offset_real_tokens_only() -> None:
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])

    positions = continuation_positions(attention_mask, prefix_len=10)

    assert positions[0].tolist() == [1, 1, 10, 11, 12]
    assert positions[1].tolist() == [10, 11, 12, 13, 14]
    assert continuation_positions(attention_mask, 0)[0].tolist() == [1, 1, 0, 1, 2]
