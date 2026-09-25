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

# Temperatures. MAX_TEMPERATURE is the HTTP contract's limit (0, 10], which
# the runtime's overrides and sample_logits both enforce. MIN_TEMPERATURE is a
# numeric floor, not a limit: below 1e-5, logits / t overflows float32 to inf
# and softmax turns it into NaN, while sampling there is already effectively
# greedy, so any smaller positive temperature is raised to it.
MIN_TEMPERATURE = 1e-5
MAX_TEMPERATURE = 10.0

# The repetition penalties apply_repetition_penalty accepts, the HTTP
# contract's [1e-4, 10]. A positive logit is divided by the penalty, so 1e-40
# overflows it to inf (then NaN); at 1e-4 a logit of 30 becomes 3e5, still
# finite. The runtime's override limits are these same constants.
MIN_REPETITION_PENALTY = 1e-4
MAX_REPETITION_PENALTY = 10.0

# What sample_logits returns for a row whose logits cannot form a distribution
# (NaN, +inf, or no finite value). It is not a valid token id, so a caller
# must check for it at the host read it already makes (the runtime's EOS check)
# and fail the request there.
UNSAMPLEABLE_TOKEN = -1


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


# sample_logits takes the contract's temperature range and floors it at
# MIN_TEMPERATURE.
_SAMPLING_TEMPERATURE_RULE = NumberRule(False, 0.0, False, MAX_TEMPERATURE)
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
    in ``(0, MAX_TEMPERATURE]`` (``ValueError`` otherwise) and is floored at
    ``MIN_TEMPERATURE``.

    A row whose logits cannot form a distribution gets ``UNSAMPLEABLE_TOKEN``
    instead of a token, on the device and without a host sync; the caller must
    check for it (see the guard below).
    """
    require_number("temperature", temperature, _SAMPLING_TEMPERATURE_RULE)
    logits = logits.clone().float()
    if token_history is not None:
        apply_repetition_penalty(logits, token_history, repetition_penalty)
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    # The guard, for the greedy and the sampling path alike. NaN or +inf in a
    # row (a model NaN, a CFG overflow), or a row with no finite logit, would
    # make softmax produce NaN and multinomial fail: on CUDA as a device-side
    # assert that poisons the whole process. A row's max is NaN if any logit
    # is (amax propagates NaN), +inf if any is +inf, and -inf if none is
    # finite, so one reduction finds all three. Such a row is replaced by
    # zeros so everything below stays well defined, and its token by
    # UNSAMPLEABLE_TOKEN. Nothing here reads a tensor on the host: the caller
    # checks the token at the read it already makes. -inf alone is fine; it is
    # how tokens are masked.
    sampleable = torch.isfinite(logits.amax(dim=-1))
    logits = torch.where(sampleable.unsqueeze(-1), logits, torch.zeros_like(logits))
    if do_sample:
        token = _draw(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        token = torch.argmax(logits, dim=-1)
    return torch.where(sampleable, token, UNSAMPLEABLE_TOKEN)


def _draw(
    logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Temperature, top-k and top-p on finite-max logits, then softmax + multinomial."""
    # temperature scaling (on raw logits, same as TemperatureLogitsWarper),
    # floored so a tiny temperature cannot overflow the logits
    logits = logits / max(temperature, MIN_TEMPERATURE)
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
