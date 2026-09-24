import threading
from types import SimpleNamespace

import torch

from models.breeze import BreezeForConditionalGeneration
from models.text_encoder_graph import TextEncoderGraphCache


def test_smallest_fitting_key_reuses_larger_token_bucket() -> None:
    keys = ((4, 32), (4, 64), (4, 96), (4, 128), (4, 160), (4, 256))

    assert TextEncoderGraphCache._smallest_fitting_key(
        keys, batch_size=4, token_length=192
    ) == (4, 256)


def test_smallest_fitting_key_does_not_cross_batch_sizes() -> None:
    keys = ((1, 256), (2, 256), (4, 160))

    assert (
        TextEncoderGraphCache._smallest_fitting_key(
            keys, batch_size=4, token_length=192
        )
        is None
    )


def _frozen_cache_with(keys: list[tuple[int, int]]) -> TextEncoderGraphCache:
    # Bypass __init__, which needs CUDA for its graph pool; the no-fit decision
    # happens before any CUDA work.
    cache = object.__new__(TextEncoderGraphCache)
    cache.token_granularity = 32
    cache._records = {key: object() for key in keys}
    cache._lock = threading.RLock()
    cache.captures = 0
    cache.replays = 0
    cache.misses = 0
    cache._frozen = True
    return cache


def test_frozen_cache_returns_none_when_no_bucket_fits() -> None:
    cache = _frozen_cache_with([(1, 32)])

    assert cache([torch.zeros(40, dtype=torch.long)]) is None
    assert cache.misses == 1
    assert cache.replays == 0


def test_frozen_cache_miss_runs_the_text_encoder_eagerly() -> None:
    encoder_calls = []

    def text_encoder(*, input_ids, attention_mask, position_ids, output_hidden_states):
        encoder_calls.append(input_ids.shape)
        return SimpleNamespace(
            last_hidden_state=input_ids.unsqueeze(-1).float(), hidden_states=None
        )

    cache = _frozen_cache_with([(1, 32)])
    model = SimpleNamespace(
        _fast_text_encoder_cudagraph=True,
        _fast_text_encoder_graph_cache=cache,
        text_encoder=text_encoder,
        text_encoder_feature_layer_idx=-1,
    )
    segment = torch.arange(1, 41, dtype=torch.long)

    hidden_states, _ = BreezeForConditionalGeneration._batched_text_encoder_forward(
        model, [segment]
    )

    assert cache.misses == 1
    assert encoder_calls == [torch.Size([1, 40])]
    assert hidden_states[0][:, 0].tolist() == segment.float().tolist()
