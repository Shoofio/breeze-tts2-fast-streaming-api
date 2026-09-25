"""Shared sampling helpers for backbone and depth decoder generation."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

# The temperature range sample_logits divides by. Below 1e-5, logits / t
# overflows float32 to inf and softmax turns it into NaN; sampling there is
# already effectively greedy, so the floor changes nothing observable. Above
# float32 max, t itself becomes inf once it meets the float32 logits, and a
# suppressed token's -inf / inf is NaN. Either NaN makes torch.multinomial
# raise. The cap is 1e4, the same as the runtime's override limit, far below
# that overflow and already a near-uniform distribution.
MIN_TEMPERATURE = 1e-5
MAX_TEMPERATURE = 1e4

# The smallest repetition penalty apply_repetition_penalty uses. A positive
# logit is divided by the penalty, so 1e-40 overflows it to inf (then NaN);
# at 1e-4 a logit of 30 becomes 3e5, still finite. The runtime rejects
# overrides below it.
MIN_REPETITION_PENALTY = 1e-4


def _require_not_nan(name: str, value: float) -> None:
    # NaN compares False against everything, so min/max clamps pass it through
    # unchanged; it must be rejected explicitly.
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN")


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits in-place and return them.

    Each distinct token in ``token_history`` is penalised once, however often
    it occurs (Hugging Face semantics). This deliberately differs from the C++
    server (``sampling.cpp``), which compounds the penalty once per occurrence
    over the whole generation so far (spec, Known Differences).

    Args:
        logits: Tensor shaped [1, 1, vocab] or [1, vocab].
        token_history: 1-D tensor of previously generated token ids.
        repetition_penalty: HF-style repetition penalty (>1.0). Values below
            ``MIN_REPETITION_PENALTY`` are raised to it; NaN raises ``ValueError``.
    """
    _require_not_nan("repetition_penalty", repetition_penalty)
    repetition_penalty = max(repetition_penalty, MIN_REPETITION_PENALTY)
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    unique_toks = token_history.unique()
    tok_logits = logits[..., unique_toks]
    logits[..., unique_toks] = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
    return logits


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    token_history: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    suppress_mask: torch.Tensor | None = None,
    suppress_tokens: Iterable[int] | None = None,
) -> torch.Tensor:
    """Sample a token from logits.

    HF-compatible order: suppress -> temperature -> top_k -> top_p -> softmax -> sample.
    Matches transformers logits_processor (TemperatureLogitsWarper, TopKLogitsWarper,
    TopPLogitsWarper) followed by softmax + multinomial. The temperature is
    clamped to ``[MIN_TEMPERATURE, MAX_TEMPERATURE]``; NaN raises ``ValueError``.
    """
    _require_not_nan("temperature", temperature)
    logits = logits.clone().float()
    if token_history is not None:
        apply_repetition_penalty(logits, token_history, repetition_penalty)
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    # temperature scaling (on raw logits, same as TemperatureLogitsWarper),
    # clamped so no caller can turn the distribution into NaN
    logits = logits / min(max(temperature, MIN_TEMPERATURE), MAX_TEMPERATURE)
    # top_k filtering (on raw logits, same as TopKLogitsWarper)
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    # top_p filtering (on raw logits, same as TopPLogitsWarper)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # HF-style shift: keep the token that pushes cumsum over top_p
        sorted_indices_to_remove = F.pad(
            sorted_indices_to_remove[..., :-1], (1, 0), value=False
        )
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter_(
            -1, sorted_indices, sorted_logits
        )
    # softmax -> multinomial (same as HF _sample)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def set_deterministic(seed=42):
    """Enable full deterministic mode and seed all RNGs for exact reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
