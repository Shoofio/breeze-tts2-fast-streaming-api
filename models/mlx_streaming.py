"""The MLX streaming runtime for Apple Silicon Macs (specs/005-mlx-mac-inference).

`MlxBreezeStreamingRuntime` implements the server's runtime seam
(contracts/runtime-seam.md) on top of mlx-audio's Breeze port (research R1), so the routes,
templates, synthesis and voice code run unchanged on a Mac.

mlx exists only on macOS arm64, and this module must still import everywhere (the room tests
run on Linux CI), so every `mlx` and `mlx_audio` import is inside the function that needs it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import torch

from breeze_infer.limits import MAX_NEW_TOKENS_CEILING
from breeze_infer.reference_audio import predicted_frames

from .cudagraph.sampling import MIN_TEMPERATURE, require_number
from .fast_streaming import (
    _POSITIVE_INTEGER_RULE,
    FastStreamingChunk,
    FastStreamingConfig,
    NoRoomError,
    PromptLength,
    _require_valid_overrides,
    prompt_length,
    select_fast_cfg,
)

# The backbone's context length. The CUDA runtime is launched with the same value, and request
# validation must match it (contracts/runtime-seam.md, `config.max_seq_len`).
MAX_SEQ_LEN = 2048


class Sampling(NamedTuple):
    """One sampler's settings, with the fields `FastBreezeStreamingRuntime._sampling_params`
    returns (models/fast_streaming.py:415)."""

    temperature: float
    top_k: int
    top_p: float
    do_sample: bool


class _PromptKeys(NamedTuple):
    """The template keys (`templates._collate_inputs`) that make up one backbone row."""

    input_ids: str
    text_ids_mask: str
    text_ids_len: str
    input_values: str


_CONDITIONAL = _PromptKeys("input_ids", "text_ids_mask", "text_ids_len", "input_values")


@dataclass(frozen=True)
class MlxCodecConfig:
    """The codec fields the server reads from `model.config.codec_config`."""

    codebook_size: int
    sampling_rate: int


@dataclass(frozen=True)
class MlxModelConfig:
    """A read-only view of the MLX checkpoint's `config.json`, with the attribute names the
    server reads from the CUDA model's config (contracts/runtime-seam.md, `model.config`).

    The values come from the same top-level JSON keys the official `BreezeConfig` takes its
    attributes from. The MLX conversion keeps them unchanged.
    """

    num_codebooks: int
    codebook_pad_token_id: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    codec_config: MlxCodecConfig

    @classmethod
    def from_config_json(cls, config: Mapping[str, Any]) -> MlxModelConfig:
        codec = config["codec_config"]
        return cls(
            num_codebooks=int(config["num_codebooks"]),
            codebook_pad_token_id=int(config["codebook_pad_token_id"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            hidden_size=int(config["hidden_size"]),
            head_dim=int(config["head_dim"]),
            codec_config=MlxCodecConfig(
                codebook_size=int(codec["codebook_size"]),
                sampling_rate=int(codec["sampling_rate"]),
            ),
        )


class MlxAudioTokenizer:
    """The server's `audio_tokenizer` (contracts/runtime-seam.md) over mlx-audio's Qwen3-TTS
    codec: reference audio in, codec codes out, as the official tokenizer returns them."""

    def __init__(self, codec: Any) -> None:
        self._codec = codec

    def get_decode_upsample_rate(self) -> int:
        return int(self._codec.decode_upsample_rate)

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, list[torch.Tensor]]:
        """Encode one waveform into `{"audio_codes": [LongTensor[frames, codebooks]]}`.

        The steps are qwen_tts's (`Qwen3TTSTokenizer._normalize_audio_inputs`, then the 12 Hz
        model's `encode`): downmix, `librosa.resample` with its defaults unless `sr` is already
        the codec's rate, encode, then keep the frames that cover the audio. That frame count
        is `reference_audio.predicted_frames`, which the server already uses to bound a
        reference before encoding it.
        """
        # librosa takes seconds to import (numba), and only reference encoding needs it.
        import librosa
        import mlx.core as mx

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        frames = predicted_frames(wav.shape[0], int(sr))
        target_sr = int(self._codec.input_sample_rate)
        if int(sr) != target_sr:
            wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=target_sr)
        # The codec takes [batch, channels, samples] and returns [batch, codebooks, frames].
        codes = self._codec.encode(mx.array(wav.astype(np.float32))[None, None, :])
        codes = np.array(codes[0, :, :frames]).T.astype(np.int64)
        return {"audio_codes": [torch.from_numpy(codes)]}


@dataclass(frozen=True)
class _MlxPrompt:
    """One backbone row of a template prompt, as MLX arrays."""

    # 1-D token ids, one array per text segment.
    text_segments: list[Any]
    # Reference codes `[1, frames, codebooks]`, or None when the prompt has no audio.
    input_values: Any | None
    # For each prompt position, the row of `_prompt_embeddings`' table it takes.
    source: Any


def _to_mlx_prompt(inputs: Mapping[str, Any], keys: _PromptKeys, config: Any) -> _MlxPrompt:
    """Convert one row of the template dict to MLX (research R4).

    This is the only place the template's torch tensors are read: torch CPU, then numpy,
    then `mx.array`. `config` is the mlx-audio model config, for the audio placeholder ids.
    """
    import mlx.core as mx

    input_ids = inputs[keys.input_ids]
    if input_ids.shape[0] != 1:
        raise ValueError(f"the MLX runtime takes one prompt per request, got {input_ids.shape[0]}")
    ids = input_ids[0].numpy()
    text_mask = inputs[keys.text_ids_mask][0].numpy().astype(bool)
    lengths = [n for n in inputs[keys.text_ids_len].numpy().tolist() if n > 0]
    text_ids = ids[text_mask]
    if sum(lengths) != text_ids.shape[0]:
        raise ValueError(
            f"text_ids_len sums to {sum(lengths)}, but text_ids_mask marks {text_ids.shape[0]} tokens"
        )
    segments = np.split(text_ids, np.cumsum(lengths)[:-1])

    values = inputs.get(keys.input_values)
    if values is None:
        audio = eos = np.zeros(ids.shape, dtype=bool)
    else:
        values = values.numpy().astype(np.int32)
        audio = ids == config.audio_token_id
        eos = ids == config.audio_eos_token_id
        if int(audio.sum()) != values.shape[1]:
            raise ValueError(
                f"the prompt has {int(audio.sum())} audio placeholders for {values.shape[1]} frames"
            )
    # As `BreezeForConditionalGeneration._merge_input_ids_with_input_values` places them: text
    # positions take the text encoder's output in order, audio placeholders the reference
    # frames in order, the audio EOS the EOS frame, and any other position stays zero (row 0).
    n_text, n_audio = text_ids.shape[0], int(audio.sum())
    source = np.zeros(ids.shape, dtype=np.int32)
    source[text_mask] = 1 + np.arange(n_text)
    source[audio] = 1 + n_text + np.arange(n_audio)
    source[eos] = 1 + n_text + n_audio
    return _MlxPrompt(
        text_segments=[mx.array(segment) for segment in segments],
        input_values=None if values is None else mx.array(values),
        source=mx.array(source),
    )


def _prompt_embeddings(model: Any, prompt: _MlxPrompt) -> Any:
    """The backbone input `[1, seq_len, hidden]` for one prompt row.

    The same steps as mlx-audio's `Model._prompt_embeddings`, which takes strings and audio
    rather than the server's token ids and codes: each text segment goes through the text
    encoder on its own (as CUDA's `_batched_text_encoder_forward` does too), and the codes
    and the EOS frame through the backbone's audio embedding.
    """
    import mlx.core as mx

    rows = [model.text_encoder_proj(model.text_encoder(ids[None]))[0] for ids in prompt.text_segments]
    if prompt.input_values is not None:
        embed = model.backbone_model.embed_tokens
        eos_frame = mx.full((1, 1, model.num_codebooks), model.config.codebook_eos_token_id, mx.int32)
        rows += [embed(prompt.input_values)[0], embed(eos_frame)[0]]
    zero = mx.zeros((1, rows[0].shape[-1]), dtype=rows[0].dtype)
    table = mx.concatenate([zero, *rows])
    return table[prompt.source][None]


def _sample(logits: Any, key: Any, sampling: Sampling) -> Any:
    """One token per row of float32 `logits`, on the GPU.

    The same steps, in the same order, as CUDA's `_sample_logits_or_sentinel` and `_draw`
    (models/cudagraph/sampling.py): temperature (floored at `MIN_TEMPERATURE`), top-k, then
    top-p with Hugging Face's shift, then a draw; `do_sample=False` is argmax.
    """
    import mlx.core as mx

    if not sampling.do_sample:
        return mx.argmax(logits, axis=-1)
    logits = logits / max(sampling.temperature, MIN_TEMPERATURE)
    if sampling.top_k > 0:
        k = min(sampling.top_k, logits.shape[-1])
        kth = mx.min(mx.topk(logits, k, axis=-1), axis=-1, keepdims=True)
        logits = mx.where(logits < kth, -mx.inf, logits)
    if sampling.top_p >= 1.0:
        return mx.random.categorical(logits, key=key)
    order = mx.argsort(-logits, axis=-1)
    ranked = mx.take_along_axis(logits, order, axis=-1)
    cumulative = mx.cumsum(mx.softmax(ranked, axis=-1), axis=-1)
    # Shifted right by one, so the token that takes the sum past top_p is kept.
    keep_first = mx.zeros(cumulative[..., :1].shape, dtype=mx.bool_)
    remove = mx.concatenate([keep_first, cumulative[..., :-1] > sampling.top_p], axis=-1)
    choice = mx.random.categorical(mx.where(remove, -mx.inf, ranked), key=key)
    return mx.take_along_axis(order, choice[:, None], axis=-1)[:, 0]


class _BackboneCache:
    """The request's backbone KV cache: one buffer per layer, `[rows, kv_heads, capacity, head_dim]`.

    The rows (the prompt, and under CFG the negative prompt) can differ in length. Each row is
    left-padded to the longest, the pad slots are masked out, and each row keeps its own RoPE
    position: the same attention as CUDA's left-padded batch with
    `position_ids = attention_mask.cumsum(-1) - 1` (fast_streaming.py `_run_prefill`).
    """

    def __init__(self, prefills: list[list[Any]], frames: int) -> None:
        import mlx.core as mx

        lengths = [cache[0].offset for cache in prefills]
        # The next slot to write, shared by every row.
        self.length = max(lengths)
        self.pad = mx.array([self.length - n for n in lengths])
        # Each row's RoPE position for the next token.
        self.positions = mx.array(lengths)
        capacity = self.length + frames
        self.keys: list[Any] = []
        self.values: list[Any] = []
        for layer in range(len(prefills[0])):
            for name, buffers in (("keys", self.keys), ("values", self.values)):
                rows = [
                    mx.pad(
                        getattr(cache[layer], name)[..., :n, :],
                        [(0, 0), (0, 0), (self.length - n, capacity - self.length), (0, 0)],
                    )
                    for cache, n in zip(prefills, lengths)
                ]
                buffers.append(mx.concatenate(rows, axis=0))


def _backbone_step(backbone: Any, cache: _BackboneCache, codes: Any) -> Any:
    """Feed one frame's codes to every row; returns the new hidden state `[rows, hidden]`.

    mlx-audio's Qwen3 blocks take one integer offset for the whole batch, so the attention is
    spelled out here with a per-row offset and pad mask; the layer weights and norms are the
    blocks' own.
    """
    import mlx.core as mx

    rows = cache.pad.shape[0]
    x = backbone.embed_tokens(mx.broadcast_to(codes[None, None, :], (rows, 1, codes.shape[0])))
    slot = cache.length
    visible = mx.arange(slot + 1)[None, :] >= cache.pad[:, None]
    mask = visible[:, None, None, :]
    for index, layer in enumerate(backbone.layers):
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(h).reshape(rows, 1, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(h).reshape(rows, 1, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = attn.v_proj(h).reshape(rows, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attn.rope(q, offset=cache.positions)
        k = attn.rope(k, offset=cache.positions)
        cache.keys[index][:, :, slot : slot + 1, :] = k
        cache.values[index][:, :, slot : slot + 1, :] = v
        out = mx.fast.scaled_dot_product_attention(
            q,
            cache.keys[index][:, :, : slot + 1],
            cache.values[index][:, :, : slot + 1],
            scale=attn.scale,
            mask=mask,
        )
        x = x + attn.o_proj(out.transpose(0, 2, 1, 3).reshape(rows, 1, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    cache.length += 1
    cache.positions = cache.positions + 1
    return backbone.norm(x)[:, -1, :]


class _Generation:
    """One request's generation state (data-model.md, "Request generation state").

    The methods only build MLX graphs. Nothing here reads a value back to the host, so the
    caller evaluates each frame once and reads it once (research R6).
    """

    def __init__(
        self,
        model: Any,
        prompts: list[_MlxPrompt],
        *,
        frames: int,
        guidance: float,
        backbone_sampling: Sampling,
        depth_sampling: Sampling,
        repetition_penalty: float,
        seed: int | None,
    ) -> None:
        import mlx.core as mx

        self._model = model
        self._guidance = guidance
        self._backbone_sampling = backbone_sampling
        self._depth_sampling = depth_sampling
        self._repetition_penalty = repetition_penalty
        # One key per request, split for every draw: the seed alone decides the random
        # stream, and mlx's global random state is never read or written.
        self._key = mx.random.key(0 if seed is None else seed)
        # Ids the backbone has sampled so far. A flag per id penalises each distinct token
        # once, as CUDA's `apply_repetition_penalty` does over the request's whole history.
        self._seen = mx.zeros((model.vocab_size + 1,), dtype=mx.bool_)

        # Each row is prefilled on its own at its exact length, then padded into one cache.
        prefills, hidden = [], []
        for prompt in prompts:
            cache = model.backbone_model.make_cache()
            embeddings = _prompt_embeddings(model, prompt)
            hidden.append(model.backbone_model(input_embeddings=embeddings, cache=cache)[:, -1, :])
            prefills.append(cache)
        self._hidden = mx.concatenate(hidden, axis=0)
        self._cache = _BackboneCache(prefills, frames)
        mx.eval(self._hidden, self._cache.keys, self._cache.values)

    def _next_key(self) -> Any:
        import mlx.core as mx

        self._key, key = mx.random.split(self._key)
        return key

    def _guided(self, logits: Any) -> Any:
        """Combine the rows `[cond, uncond]` as `uncond + g·(cond − uncond)`, in float32 as
        CUDA does (`BackboneGraph`, `DepthDecoderGraph._cfg_sample`). One row passes through."""
        import mlx.core as mx

        logits = logits.astype(mx.float32)
        if logits.shape[0] == 1:
            return logits
        cond, uncond = logits[:1], logits[1:]
        return uncond + self._guidance * (cond - uncond)

    def frame(self) -> Any:
        """The next frame's codes, `[num_codebooks]` int32: the backbone's token (EOS when it
        equals `vocab_size`), then the depth decoder's."""
        import mlx.core as mx

        model = self._model
        logits = self._guided(model.lm_head(self._hidden))
        if self._repetition_penalty != 1.0:
            penalty = self._repetition_penalty
            penalised = mx.where(logits > 0, logits / penalty, logits * penalty)
            logits = mx.where(self._seen, penalised, logits)
        # The reserved codec ids are masked after the penalty, as on CUDA. EOS stays sampleable.
        first = _sample(
            model._mask_reserved_codec_logits(logits), self._next_key(), self._backbone_sampling
        )
        self._seen = self._seen | (mx.arange(self._seen.shape[0]) == first)
        return mx.concatenate([first, *self._depth_codes(first)]).astype(mx.int32)

    def _depth_codes(self, first: Any) -> list[Any]:
        """Codebooks 1 to `num_codebooks - 1` from the depth decoder, with its own KV cache.

        Step 0 feeds `[backbone state, codebook 0]` and each later step one code: the same
        positions and causal mask as CUDA's `DepthDecoderGraph` and mlx-audio's full re-run
        (`Model._depth_tokens`). Under CFG both rows run as one batch of 2.
        """
        import mlx.core as mx
        from mlx_audio.lm.models.cache import KVCache

        depth = self._model.depth_decoder
        decoder = depth.model
        rows = self._hidden.shape[0]
        hidden = self._hidden
        if decoder.backbone_hidden_state_projector is not None:
            hidden = decoder.backbone_hidden_state_projector(hidden)

        def embed(code: Any, codebook: int) -> Any:
            embedded = decoder.embed_tokens(code + codebook * decoder.vocab_size)
            return mx.broadcast_to(embedded[None], (rows, 1, embedded.shape[-1]))

        x = decoder.inputs_embeds_projector(
            mx.concatenate([hidden[:, None, :], embed(first, 0)], axis=1)
        )
        caches = [KVCache() for _ in decoder.layers]
        mask = "causal"
        codes = []
        last = self._model.num_codebooks - 1
        for codebook in range(1, last + 1):
            for layer, cache in zip(decoder.layers, caches):
                x = layer(x, mask, cache)
            head = depth.codebooks_head.weight[codebook - 1]
            logits = self._guided(decoder.norm(x)[:, -1, :] @ head)
            code = _sample(
                self._model._mask_reserved_codec_logits(logits),
                self._next_key(),
                self._depth_sampling,
            )
            codes.append(code)
            if codebook < last:
                x = decoder.inputs_embeds_projector(embed(code, codebook))
                mask = None
        return codes

    def advance(self, codes: Any) -> None:
        """Run the backbone on `codes`, the frame just made, for the next frame."""
        self._hidden = _backbone_step(self._model.backbone_model, self._cache, codes)


class MlxBreezeStreamingRuntime:
    """The runtime seam (contracts/runtime-seam.md) on MLX.

    `load_mlx_runtime` builds it with the loaded mlx-audio model, its codec and the tokenizer.
    The room rules read only `config` and `generation_config`'s `max_new_tokens`, so the
    model-free tests build it with `None` for the loaded parts and no sampling defaults.
    """

    # The KV cache is bf16 at both precisions (mxfp8 quantizes weights, not activations), so
    # the voice prefix cache's byte estimate is the same as on CUDA.
    dtype = torch.bfloat16
    # MLX warms up through its own branch in model loading, not the CUDA graph warmup.
    fast_enabled = False
    # Frames per yielded chunk, as on the CUDA path without --fast-codec (research R6).
    codec_chunk_frames = 2

    def __init__(
        self,
        *,
        mlx_model: Any,
        codec: Any,
        tokenizer: Any,
        model_config: MlxModelConfig | None,
        generation_config: Mapping[str, Any],
        backbone_sampling: Sampling | None = None,
        depth_sampling: Sampling | None = None,
    ) -> None:
        self._mlx_model = mlx_model
        self._codec = codec
        self._backbone_sampling = backbone_sampling
        self._depth_sampling = depth_sampling
        self.audio_tokenizer = MlxAudioTokenizer(codec)
        self.tokenizer = tokenizer
        # The server only reads `.config` and uses `.device` as the `.to()` target for template
        # tensors, which are converted to MLX per request (research R4).
        self.model = SimpleNamespace(config=model_config, device="cpu")
        self.config = FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS_CEILING, max_seq_len=MAX_SEQ_LEN
        )
        default = generation_config.get("max_new_tokens")
        # Validated at load, as the CUDA runtime does: it sizes every request that sets none.
        if default is not None:
            require_number("generation_config.max_new_tokens", default, _POSITIVE_INTEGER_RULE)
        self._default_max_new_tokens = default

    @property
    def sample_rate(self) -> int:
        return self.model.config.codec_config.sampling_rate

    def frame_cap(self, requested: int | None) -> int:
        """Frames a request may generate, before any context limit: `requested`, or the
        checkpoint's default when it is `None`, clamped to the ceiling. The same rule as
        `FastBreezeStreamingRuntime.frame_cap`."""
        if requested is None:
            requested = self._default_max_new_tokens or self.config.max_new_tokens
        return min(int(requested), self.config.max_new_tokens)

    def max_new_tokens_room(
        self, requested: int | None, inputs: dict[str, Any], *, prefix_len: int = 0
    ) -> int:
        """Frames one request for `inputs` will produce at most; `<= 0` means none."""
        return self.room_for_length(requested, prompt_length(inputs), prefix_len=prefix_len)

    def room_for_length(
        self, requested: int | None, length: PromptLength, *, prefix_len: int = 0
    ) -> int:
        """`max_new_tokens_room` for a prompt known only by its `PromptLength`.

        MLX has no prefill buckets, so the prefill is always the exact length: CUDA's rule
        without --fast-backbone-prefill (research R4). The decode loop stops once
        `prefill_len + step >= max_seq_len - 1`.
        """
        _require_valid_overrides(max_new_tokens=requested)
        prefill_len = prefix_len + length.seq_len
        return min(self.frame_cap(requested), self.config.max_seq_len - prefill_len - 1)

    def iter_audio_chunks(
        self,
        inputs: dict[str, Any],
        *,
        request_id: str | None = None,
        seed: int | None = None,
        token_observer: Callable[[torch.Tensor], None] | None = None,
        prefix: Any | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[FastStreamingChunk]:
        """Stream one request's audio (contracts/runtime-seam.md, `iter_audio_chunks`).

        Nothing runs until the first `next()`. Invalid overrides raise `ValueError` there,
        and a prompt with no room raises `NoRoomError`. The sampling overrides apply to the
        backbone only; the depth decoder keeps the model defaults, as on CUDA. Generation
        stops at EOS or after `max_new_tokens_room` frames. `request_id` is part of the seam
        and unused here: the MLX codec serves one request at a time.
        """
        _require_valid_overrides(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if prefix is not None:
            raise NotImplementedError("the MLX runtime does not take reference prefixes yet")
        cfg = select_fast_cfg(inputs)
        if cfg.mode != "no_cfg" or cfg.use_negative_as_main:
            raise NotImplementedError("the MLX runtime does not run CFG yet")
        frames = self.max_new_tokens_room(max_new_tokens, inputs)
        if frames <= 0:
            raise NoRoomError(
                f"prompt leaves no room to generate in the {self.config.max_seq_len}-token context"
            )
        defaults = self._backbone_sampling
        backbone_sampling = defaults._replace(
            temperature=defaults.temperature if temperature is None else float(temperature),
            top_k=defaults.top_k if top_k is None else int(top_k),
            top_p=defaults.top_p if top_p is None else float(top_p),
        )
        penalty = (
            self.config.repetition_penalty if repetition_penalty is None else float(repetition_penalty)
        )

        model = self._mlx_model
        decoder = self._codec.decoder
        decoder.reset_streaming_state()
        try:
            generation = _Generation(
                model,
                [_to_mlx_prompt(inputs, _CONDITIONAL, model.config)],
                frames=frames,
                guidance=cfg.guidance_scale,
                backbone_sampling=backbone_sampling,
                depth_sampling=self._depth_sampling,
                repetition_penalty=penalty,
                seed=seed,
            )
            yield from self._stream(generation, frames, token_observer)
        finally:
            # The codec's convolution buffers and KV cache live on the shared decoder. The rest
            # of the request's state lives in this generator and goes with it, also on close().
            decoder.reset_streaming_state()

    def _stream(
        self,
        generation: _Generation,
        frames: int,
        token_observer: Callable[[torch.Tensor], None] | None,
    ) -> Iterator[FastStreamingChunk]:
        """The frame loop (research R6).

        Frame n+1 is queued on the GPU before frame n is read, so the GPU keeps working while
        the host reads a frame, observes it and queues the codec. Each frame is evaluated once
        and read once. A chunk's audio is copied out one frame after it is queued, when it is
        already done. A frame queued after EOS is never read.
        """
        import mlx.core as mx

        eos = self._mlx_model.vocab_size
        pad = self.model.config.codebook_pad_token_id
        pending: list[Any] = []
        queued: tuple[Any, int] | None = None
        codes = generation.frame()
        mx.async_eval(codes)
        for step in range(frames):
            last = step == frames - 1
            if not last:
                generation.advance(codes)
                next_codes = generation.frame()
                mx.async_eval(next_codes)
            # The frame's one host read: the EOS check, the observer and the pad check use it.
            host = np.array(codes)
            if int(host[0]) == eos:
                break
            if queued is not None:
                yield self._chunk(*queued, is_final=False)
                queued = None
            if token_observer is not None:
                token_observer(torch.from_numpy(host.astype(np.int64)))
            # An all-pad frame is observed but not decoded, as on CUDA (`_frame_flags`).
            if not (host == pad).all():
                pending.append(codes)
            if pending and (len(pending) == self.codec_chunk_frames or last):
                queued = self._decode(pending)
                pending = []
            if not last:
                codes = next_codes
        if pending:
            queued = self._decode(pending)
        if queued is not None:
            yield self._chunk(*queued, is_final=True)

    def _decode(self, frames: list[Any]) -> tuple[Any, int]:
        """Queue the codec's incremental decode of `frames`; returns the lazy audio and the
        frame count."""
        import mlx.core as mx

        # streaming_step takes [batch, codebooks, frames].
        audio = self._codec.decoder.streaming_step(mx.stack(frames, axis=1)[None])
        mx.async_eval(audio)
        return audio, len(frames)

    def _chunk(self, audio: Any, frames: int, *, is_final: bool) -> FastStreamingChunk:
        return FastStreamingChunk(
            audio=np.array(audio, dtype=np.float32).reshape(-1),
            sample_rate=self.sample_rate,
            codec_frames=frames,
            is_final=is_final,
            timing={},
        )


def _sampling(generation_config: Any) -> Sampling:
    """A sampler's settings from a `GenerationConfig`, with the fallbacks of
    `FastBreezeStreamingRuntime._sampling_params` (models/fast_streaming.py:415-427). The
    server's `FastStreamingConfig` sets no sampling overrides (`model_loading.streaming_config`),
    so the config's values decide."""

    def value(name: str, default: Any) -> Any:
        found = getattr(generation_config, name, default)
        return default if found is None else found

    return Sampling(
        temperature=float(value("temperature", 1.0)),
        top_k=int(value("top_k", 0)),
        top_p=float(value("top_p", 1.0)),
        do_sample=bool(value("do_sample", True)),
    )


def _load_sampling_defaults(path: Path) -> tuple[Sampling, Sampling]:
    """The backbone's and the depth decoder's default sampling, sourced as on CUDA.

    On CUDA the model's `generation_config` comes from the checkpoint's
    `generation_config.json`, the depth decoder's starts from transformers' defaults, and
    `model_loading.load_model` then applies `update_generation_config_for_breeze`
    (breeze_infer/model_loading.py:111, breeze_infer/runtime.py:45-70). That call's built-in
    values override the file for every sampling setting of both. The same function is applied
    here to the same starting configs, so the two backends cannot drift apart.
    """
    from transformers import GenerationConfig

    from breeze_infer.runtime import update_generation_config_for_breeze

    model = SimpleNamespace(
        generation_config=GenerationConfig.from_pretrained(path),
        depth_decoder=SimpleNamespace(generation_config=GenerationConfig()),
    )
    update_generation_config_for_breeze(model)
    return _sampling(model.generation_config), _sampling(model.depth_decoder.generation_config)


def load_mlx_runtime(path: Path) -> MlxBreezeStreamingRuntime:
    """Load an MLX Breeze checkpoint (bf16 or mxfp8) into a runtime."""
    from mlx_audio.tts.utils import load
    from transformers import AutoTokenizer

    # mlx-audio's Breeze loader validates the weights and, in its post-load hook, loads the
    # codec from `path / "audio_tokenizer"` with its strict codec-weights check.
    mlx_model = load(path)
    # AutoTokenizer loads the MLX checkpoint directly with transformers 4.57.3, which this
    # backend pins (research R2): `tokenizer_config.json` names `GemmaTokenizerFast`, so the
    # unknown `breeze_tts` model type never comes into it. The files are the official ones
    # (research R3), loaded the way `breeze_infer.runtime` loads them on CUDA. mlx-audio's own
    # `mlx_model.tokenizer` is not used.
    tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
    backbone_sampling, depth_sampling = _load_sampling_defaults(path)
    return MlxBreezeStreamingRuntime(
        mlx_model=mlx_model,
        codec=mlx_model.audio_tokenizer,
        tokenizer=tokenizer,
        model_config=MlxModelConfig.from_config_json(config),
        generation_config=generation_config,
        backbone_sampling=backbone_sampling,
        depth_sampling=depth_sampling,
    )
