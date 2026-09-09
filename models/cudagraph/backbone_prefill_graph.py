"""Static-shape CUDA Graph buckets for backbone prefill.

One graph per ``(batch, bucket)``. The prompt (or, with a cached reference
prefix, only its suffix) is right-aligned in the bucket and written to cache
slots ``[prefix_len, prefix_len + bucket)``; every query may attend to the
prefix already in ``[0, prefix_len)``. ``prefix_len`` is zero for a full
prefill and varies per replay otherwise. It only changes the *values* of the
static position, cache-position and mask buffers, never their shapes, so a
single captured graph serves both cases.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


@dataclass
class BackbonePrefillOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    attention_mask: torch.Tensor
    prefill_len: int


@dataclass
class _PrefillRecord:
    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    causal_mask: torch.Tensor
    cache_position: torch.Tensor
    hidden_states: torch.Tensor
    logits: torch.Tensor


def continuation_allowed_keys(
    attention_mask: torch.Tensor, key_len: int, prefix_len: int
) -> torch.Tensor:
    """Which cache keys each query may attend to, as ``bool[B, Q, key_len]``.

    Queries are the ``Q`` bucket positions written to cache slots
    ``[prefix_len, prefix_len + Q)``. Every query may see the whole prefix
    ``[0, prefix_len)``; within the bucket it sees causal, unpadded keys only.
    With ``prefix_len == 0`` this is the ordinary left-padded causal mask.
    """
    batch_size, query_len = attention_mask.shape
    if prefix_len < 0:
        raise ValueError("prefix_len must be >= 0")
    if prefix_len + query_len > key_len:
        raise ValueError(
            f"prefix_len {prefix_len} + bucket {query_len} exceeds key_len {key_len}"
        )
    device = attention_mask.device
    allowed = torch.zeros((batch_size, query_len, key_len), dtype=torch.bool, device=device)
    if prefix_len > 0:
        allowed[:, :, :prefix_len] = True
    query_idx = torch.arange(query_len, device=device).view(1, query_len, 1)
    key_idx = torch.arange(query_len, device=device).view(1, 1, query_len)
    within_bucket = (key_idx <= query_idx) & attention_mask.to(torch.bool)[:, None, :]
    allowed[:, :, prefix_len : prefix_len + query_len] = within_bucket
    return allowed


def continuation_positions(attention_mask: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """RoPE positions for a (left-padded) bucket that follows ``prefix_len`` tokens."""
    positions = attention_mask.long().cumsum(-1) - 1 + int(prefix_len)
    positions.masked_fill_(attention_mask == 0, 1)
    return positions


class BackbonePrefillGraphCache:
    """Capture backbone prefill + lm_head while writing the decode StaticCache."""

    def __init__(self, backbone_graph, *, token_granularity: int = 32) -> None:
        if token_granularity <= 0:
            raise ValueError("token_granularity must be > 0")
        self.backbone_graph = backbone_graph
        self.model = backbone_graph.model
        self.lm_head = backbone_graph.lm_head
        self.device = torch.device(backbone_graph.device)
        self.dtype = backbone_graph.dtype
        self.token_granularity = int(token_granularity)
        self._records: dict[tuple[int, int], _PrefillRecord] = {}
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._lock = threading.RLock()
        self.captures = 0
        self.replays = 0
        self._frozen = False

    def _bucket(self, length: int) -> int:
        return (
            (int(length) + self.token_granularity - 1) // self.token_granularity
        ) * self.token_granularity

    @staticmethod
    def _copy_inputs(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        static_embeds: torch.Tensor,
        static_mask: torch.Tensor,
        static_positions: torch.Tensor,
        static_cache_position: torch.Tensor,
        prefix_len: int,
    ) -> None:
        seq_len = int(inputs_embeds.shape[1])
        pad_len = int(static_embeds.shape[1]) - seq_len
        static_embeds.zero_()
        static_mask.zero_()
        static_embeds[:, pad_len:].copy_(inputs_embeds)
        static_mask[:, pad_len:].copy_(attention_mask)
        static_positions.copy_(continuation_positions(static_mask, prefix_len))
        static_cache_position.copy_(
            torch.arange(
                prefix_len,
                prefix_len + int(static_mask.shape[1]),
                device=static_cache_position.device,
                dtype=torch.long,
            )
        )

    @staticmethod
    def _update_causal_mask(
        attention_mask: torch.Tensor, causal_mask: torch.Tensor, prefix_len: int = 0
    ) -> None:
        allowed = continuation_allowed_keys(
            attention_mask, int(causal_mask.shape[-1]), prefix_len
        )
        causal_mask.fill_(torch.finfo(causal_mask.dtype).min)
        causal_mask[:, 0].masked_fill_(allowed, 0.0)

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.backbone_graph.static_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        hidden_states = output.last_hidden_state
        logits = self.lm_head(hidden_states[:, -1, :].float()).float()
        return hidden_states, logits

    @torch.inference_mode()
    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prefix_len: int = 0,
    ) -> BackbonePrefillOutput:
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        if batch_size != self.backbone_graph.batch_size:
            raise ValueError(
                f"prefill batch={batch_size} does not match decode graph "
                f"batch={self.backbone_graph.batch_size}"
            )
        prefix_len = int(prefix_len)
        bucket_len = self._bucket(int(seq_len))
        if prefix_len + bucket_len > self.backbone_graph.max_seq_len:
            raise RuntimeError(
                f"prefill bucket {bucket_len} after prefix {prefix_len} exceeds "
                f"max_seq_len {self.backbone_graph.max_seq_len}"
            )
        key = (int(batch_size), bucket_len)

        with self._lock:
            record = self._records.get(key)
            if record is None:
                if self._frozen:
                    raise RuntimeError(
                        f"backbone prefill CUDA graph {key} was not declared in the warmup profile"
                    )
                static_embeds = torch.zeros(
                    (batch_size, bucket_len, hidden_size),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )
                static_mask = torch.zeros(
                    (batch_size, bucket_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                static_positions = torch.ones_like(static_mask, dtype=torch.long)
                static_causal_mask = torch.empty(
                    (batch_size, 1, bucket_len, self.backbone_graph.max_seq_len),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )
                static_cache_position = torch.arange(
                    bucket_len, device=inputs_embeds.device, dtype=torch.long
                )
                self._copy_inputs(
                    inputs_embeds,
                    attention_mask,
                    static_embeds,
                    static_mask,
                    static_positions,
                    static_cache_position,
                    prefix_len,
                )
                self._update_causal_mask(static_mask, static_causal_mask, prefix_len)

                capture_stream = torch.cuda.Stream(device=inputs_embeds.device)
                capture_stream.wait_stream(
                    torch.cuda.current_stream(inputs_embeds.device)
                )
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        hidden_states, logits = self._forward(
                            static_embeds,
                            static_causal_mask,
                            static_positions,
                            static_cache_position,
                        )
                capture_stream.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    stream=capture_stream,
                    pool=self._graph_pool,
                    capture_error_mode="thread_local",
                ):
                    hidden_states, logits = self._forward(
                        static_embeds,
                        static_causal_mask,
                        static_positions,
                        static_cache_position,
                    )
                record = _PrefillRecord(
                    graph=graph,
                    stream=capture_stream,
                    inputs_embeds=static_embeds,
                    attention_mask=static_mask,
                    position_ids=static_positions,
                    causal_mask=static_causal_mask,
                    cache_position=static_cache_position,
                    hidden_states=hidden_states,
                    logits=logits,
                )
                self._records[key] = record
                self.captures += 1

            self._copy_inputs(
                inputs_embeds,
                attention_mask,
                record.inputs_embeds,
                record.attention_mask,
                record.position_ids,
                record.cache_position,
                prefix_len,
            )
            self._update_causal_mask(record.attention_mask, record.causal_mask, prefix_len)
            current_stream = torch.cuda.current_stream(inputs_embeds.device)
            record.stream.wait_stream(current_stream)
            record.graph.replay()
            current_stream.wait_stream(record.stream)
            prefill_len = prefix_len + bucket_len
            self.backbone_graph.finish_direct_prefill(prefill_len)
            self.replays += 1
            return BackbonePrefillOutput(
                hidden_states=record.hidden_states,
                logits=record.logits,
                attention_mask=record.attention_mask,
                prefill_len=prefill_len,
            )

    @property
    def graph_keys(self) -> tuple[tuple[int, int], ...]:
        """Captured buckets as ``(batch, bucket)``."""
        return tuple(sorted(self._records))

    def has_bucket(self, batch_size: int, seq_len: int, prefix_len: int = 0) -> bool:
        """Whether a replay for ``seq_len`` tokens after ``prefix_len`` can use a graph."""
        bucket_len = self._bucket(seq_len)
        return (int(batch_size), bucket_len) in self._records and (
            int(prefix_len) + bucket_len <= self.backbone_graph.max_seq_len
        )

    @torch.inference_mode()
    def warmup_graph(self, *, branch_batch_size: int, sequence_length: int) -> None:
        if self._frozen:
            raise RuntimeError("backbone prefill CUDA graph cache is already frozen")
        inputs_embeds, attention_mask = self._warmup_inputs(
            branch_batch_size, sequence_length
        )
        self(inputs_embeds, attention_mask)

    def _warmup_inputs(
        self, branch_batch_size: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = torch.zeros(
            branch_batch_size,
            sequence_length,
            int(self.backbone_graph.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        attention_mask = torch.ones(
            branch_batch_size,
            sequence_length,
            dtype=torch.long,
            device=self.device,
        )
        return inputs_embeds, attention_mask

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True
