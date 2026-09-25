"""Shared sampling helpers for backbone and depth decoder generation."""

from __future__ import annotations

import math
import numbers
import os
import random
from collections.abc import Iterable
from typing import Any, NamedTuple

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

# The repetition penalties apply_repetition_penalty accepts. A positive logit
# is divided by the penalty, so 1e-40 overflows it to inf (then NaN); at 1e-4 a
# logit of 30 becomes 3e5, still finite. The upper bound is far past any useful
# setting and keeps absurd values a caller error rather than a numerics
# question. The runtime's override limits are these same constants.
MIN_REPETITION_PENALTY = 1e-4
MAX_REPETITION_PENALTY = 1e4


class NumberRule(NamedTuple):
    """What a sampling setting accepts: an integer or a real, within a range."""

    integer: bool
    minimum: float
    minimum_inclusive: bool
    maximum: float = math.inf

    def describe(self) -> str:
        kind = "an integer" if self.integer else "a finite number"
        low = "[" if self.minimum_inclusive else "("
        high = "inf)" if self.maximum == math.inf else f"{self.maximum:g}]"
        return f"{kind} in {low}{self.minimum:g}, {high}"


def is_valid_number(value: Any, rule: NumberRule) -> bool:
    """Whether ``value`` satisfies ``rule``; never raises.

    Any ``numbers.Integral``/``numbers.Real`` counts (so numpy scalars do), but
    not a bool. NaN fails both range comparisons. The range is checked before
    ``math.isfinite``, which raises OverflowError for an int too large for a
    float (10**400) while the comparisons are exact.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, numbers.Integral if rule.integer else numbers.Real):
        return False
    above = value >= rule.minimum if rule.minimum_inclusive else value > rule.minimum
    if not (above and value <= rule.maximum):
        return False
    if rule.integer:
        return True
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def require_number(name: str, value: Any, rule: NumberRule, *, suffix: str = "") -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value`` satisfies ``rule``.

    The one number check for sampling settings, here and in the runtime.
    ``suffix`` extends the expectation (such as " or None").
    """
    if not is_valid_number(value, rule):
        raise ValueError(f"{name} must be {rule.describe()}{suffix}, got {value!r}")


# sample_logits takes any finite positive temperature and clamps it; the
# runtime's per-request override rule is the stricter (0, MAX_TEMPERATURE].
_SAMPLING_TEMPERATURE_RULE = NumberRule(False, 0.0, False)
REPETITION_PENALTY_RULE = NumberRule(
    False, MIN_REPETITION_PENALTY, True, MAX_REPETITION_PENALTY
)


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float | None,
) -> torch.Tensor:
    """Apply repetition penalty to logits in-place and return them.

    Each distinct token in ``token_history`` is penalised once, however often
    it occurs (Hugging Face semantics). This deliberately differs from the C++
    server (``sampling.cpp``), which compounds the penalty once per occurrence
    over the whole generation so far (spec, Known Differences).

    Args:
        logits: Tensor shaped [1, 1, vocab] or [1, vocab].
        token_history: 1-D tensor of previously generated token ids.
        repetition_penalty: HF-style repetition penalty (>1.0 discourages
            repeats), in ``[MIN_REPETITION_PENALTY, MAX_REPETITION_PENALTY]``;
            anything else, NaN included, raises ``ValueError``. ``None`` disables it.
    """
    if repetition_penalty is None:
        return logits
    require_number("repetition_penalty", repetition_penalty, REPETITION_PENALTY_RULE)
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
    repetition_penalty: float | None = None,
    suppress_mask: torch.Tensor | None = None,
    suppress_tokens: Iterable[int] | None = None,
) -> torch.Tensor:
    """Sample a token from logits.

    HF-compatible order: suppress -> temperature -> top_k -> top_p -> softmax -> sample.
    Matches transformers logits_processor (TemperatureLogitsWarper, TopKLogitsWarper,
    TopPLogitsWarper) followed by softmax + multinomial. The temperature must be
    finite and > 0 (``ValueError`` otherwise) and is clamped to
    ``[MIN_TEMPERATURE, MAX_TEMPERATURE]``. Logits that cannot form a
    distribution raise ``ValueError`` before softmax (see the guard below).
    """
    require_number("temperature", temperature, _SAMPLING_TEMPERATURE_RULE)
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
    # The one guard before softmax. NaN or +inf anywhere (a model NaN, a CFG
    # overflow), or a row with no finite logit, makes softmax produce NaN, and
    # multinomial then fails: on CUDA as a device-side assert that poisons the
    # whole process. Raising ValueError instead fails only this request, and
    # the caller's stream error path handles it. -inf is fine: it is how
    # tokens are masked. A row's max is NaN if any logit is (amax propagates
    # NaN), +inf if any is +inf, and -inf if none is finite, so one reduction
    # covers all three; the check costs one host sync per call.
    if not bool(torch.isfinite(logits.amax(dim=-1)).all()):
        raise ValueError(
            "logits contain NaN or +inf, or no finite value; cannot sample"
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
