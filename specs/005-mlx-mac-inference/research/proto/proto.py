# ruff: noqa  (reference-only prototype; see README.md in this directory)
"""Task 2/3: prototype Breeze frame loop with cumulative optimisations.

Levels (cumulative):
  stock : unmodified Model.generate()            (frames recorded via an instance-level wrapper)
  a     : + depth decoder KV cache               (CFG still sequential, .item() per token)
  b     : + CFG as batch 2 (backbone and depth)  (still .item() per token)
  c     : + no per-token host syncs, lazy graph per frame, one-frame-lagged EOS check,
            async_eval pipelining, codec every 2 frames
  d     : + mx.compile of the whole frame step (head sampling + 15 depth steps + backbone step)
            with a fixed-capacity backbone KV buffer so shapes never change
  e     : + codec streaming decode on a second GPU stream (extra experiment)

Nothing in mlx-audio is modified; we only reuse its modules' weights/sub-layers.

usage: proto.py <8bit|bf16> <text-name> <level> [--cfg] [--greedy] [--max N]
               [--save-frames F.json] [--wav F.wav]
Prints one JSON line.
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.sample_utils import make_sampler

from common import CFG_SCALE, INSTRUCTION, PROTO, load_model, log, make_fp32, text_by_name, write_wav

STREAM_FRAMES = 2  # codec chunk: 0.16 s at 12.5 Hz, as in the gate


# --------------------------------------------------------------------------- sampling
class Sampler:
    """Same ops as Model._sample (mask reserved ids, slice, log_softmax, mlx-lm sampler)
    but returns an mx.array instead of calling .item()."""

    def __init__(self, model, temperature, top_p, top_k):
        self.V = model.vocab_size
        self.codec_V = model.config.codec_vocab_size
        self.model = model
        self.samplers = {}
        for allow_eos in (False, True):
            valid = self.V + 1 if allow_eos else self.V
            k = min(top_k, valid) if top_k else 0
            if k == valid:
                k = 0
            self.samplers[allow_eos] = make_sampler(temp=temperature, top_p=top_p, top_k=k)

    def __call__(self, logits, allow_eos=False):
        valid = self.V + 1 if allow_eos else self.V
        logits = self.model._mask_reserved_codec_logits(logits)[..., :valid]
        return self.samplers[allow_eos](nn.log_softmax(logits, axis=-1))  # shape [1]


# --------------------------------------------------------------------------- depth decoder
def _attn(attn, x, kv, offset, mask):
    """llama/qwen3-style attention with an explicit KV list; returns (out, (k, v))."""
    B, L, _ = x.shape
    q = attn.q_proj(x).reshape(B, L, attn.n_heads, -1)
    k = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)
    v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    q = attn.rope(q.transpose(0, 2, 1, 3), offset=offset)
    k = attn.rope(k.transpose(0, 2, 1, 3), offset=offset)
    if kv is not None:
        k = mx.concatenate([kv[0], k], axis=2)
        v = mx.concatenate([kv[1], v], axis=2)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    return attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v)


def _block(layer, x, kv, offset, mask):
    r, kv = _attn(layer.self_attn, layer.input_layernorm(x), kv, offset, mask)
    h = x + r
    return h + layer.mlp(layer.post_attention_layernorm(h)), kv


class CachedDepth:
    """Depth decoder with a KV cache: step 0 feeds [backbone_state, codebook0] (L=2),
    each later step feeds one token (L=1). Mathematically identical to the stock
    re-run of the growing sequence (same RoPE positions, causal mask)."""

    def __init__(self, model):
        self.dd = model.depth_decoder
        self.dm = model.depth_decoder.model
        self.vocab = self.dm.vocab_size

    def start(self, hidden, cb0):
        """hidden [B,H]; cb0 [1] int -> logits for codebook 1, plus kv state."""
        dm = self.dm
        B = hidden.shape[0]
        h0 = hidden
        if dm.backbone_hidden_state_projector is not None:
            h0 = dm.backbone_hidden_state_projector(h0)
        e1 = dm.embed_tokens(cb0)  # codebook 0 offset is 0
        e1 = mx.broadcast_to(e1[None], (B, 1, e1.shape[-1])) if e1.ndim == 2 else e1
        x = dm.inputs_embeds_projector(mx.concatenate([h0[:, None, :], e1], axis=1))
        kvs = []
        for layer in dm.layers:
            x, kv = _block(layer, x, None, 0, "causal")
            kvs.append(kv)
        h = dm.norm(x)[:, -1, :]
        return h @ self.dd.codebooks_head.weight[0], kvs

    def step(self, tok, n, kvs):
        """tok [1] = codebook n (1..14) -> logits for codebook n+1."""
        dm = self.dm
        B = kvs[0][0].shape[0]
        e = dm.embed_tokens(tok + n * self.vocab)  # [1, E]
        e = mx.broadcast_to(e[None], (B, 1, e.shape[-1]))
        x = dm.inputs_embeds_projector(e)
        new = []
        for layer, kv in zip(dm.layers, kvs):
            x, kv = _block(layer, x, kv, n + 1, None)
            new.append(kv)
        h = dm.norm(x)[:, -1, :]
        return h @ self.dd.codebooks_head.weight[n], new


def cfg_combine(logits, scale, batched):
    """Batched rows are [cond, uncond]."""
    if not batched:
        return logits
    c, u = logits[0:1], logits[1:2]
    return u + scale * (c - u)


# --------------------------------------------------------------------------- backbone buffer
class BackboneBuffer:
    """Fixed-capacity KV buffer for the backbone, batch B (rows: cond[, uncond]).

    Rows may have different prompt lengths: row r is left-padded by pad[r] slots and
    rotated with its own RoPE offset (per-row offsets), so positions match stock exactly.
    """

    def __init__(self, model, caches, capacity):
        bb = model.backbone_model
        self.bb = bb
        lens = [c[0].offset for c in caches]
        self.Lmax = max(lens)
        self.pad = mx.array([self.Lmax - n for n in lens], dtype=mx.int32)
        self.lens = mx.array(lens, dtype=mx.int32)
        self.cap = capacity
        self.K, self.V = [], []
        for i in range(len(bb.layers)):
            ks, vs = [], []
            for c, n in zip(caches, lens):
                k = c[i].keys[..., :n, :]
                v = c[i].values[..., :n, :]
                padw = [(0, 0), (0, 0), (self.Lmax - n, capacity - self.Lmax), (0, 0)]
                ks.append(mx.pad(k, padw))
                vs.append(mx.pad(v, padw))
            self.K.append(mx.concatenate(ks, axis=0))
            self.V.append(mx.concatenate(vs, axis=0))
        mx.eval(self.K, self.V)
        self.slot = self.Lmax  # next write slot (shared by all rows)
        self.arange = mx.arange(capacity)


def backbone_step_fn(bb, frame, slot, rope_off, pad, Ks, Vs, attend_len, B):
    """One backbone decode step on the buffer. frame [16] int; slot scalar int array;
    rope_off [B] int; pad [B] int. attend_len: Python int (static) number of key slots
    to attend (slot+1 for the uncompiled path, capacity for the compiled path).
    Returns hidden [B,H] and new K/V lists."""
    x = bb.embed_tokens(mx.broadcast_to(frame[None, None, :], (B, 1, frame.shape[0])))
    ar = mx.arange(attend_len)
    valid = (ar[None, :] <= slot) & (ar[None, :] >= pad[:, None])  # [B, attend_len]
    mask = valid[:, None, None, :]
    start = mx.concatenate([mx.zeros((2,), mx.int32), slot.reshape(1).astype(mx.int32), mx.zeros((1,), mx.int32)])
    newK, newV = [], []
    for layer, K, V in zip(bb.layers, Ks, Vs):
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(h).reshape(B, 1, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(h).reshape(B, 1, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = attn.v_proj(h).reshape(B, 1, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attn.rope(q, offset=rope_off)
        k = attn.rope(k, offset=rope_off)
        K = mx.slice_update(K, k, start, axes=(0, 1, 2, 3))
        V = mx.slice_update(V, v, start, axes=(0, 1, 2, 3))
        newK.append(K)
        newV.append(V)
        kk, vv = (K, V) if attend_len == K.shape[2] else (K[:, :, :attend_len], V[:, :, :attend_len])
        o = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=attn.scale, mask=mask)
        x = x + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, 1, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return bb.norm(x)[:, -1, :], newK, newV


# --------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, model, level, cfg, temperature, top_p, top_k, max_tokens):
        self.m = model
        self.level = level
        self.cfg = cfg
        self.scale = CFG_SCALE
        self.batched = cfg and level in ("b", "c", "d", "e")
        self.B = 2 if self.batched else 1
        self.sampler = Sampler(model, temperature, top_p, top_k)
        self.depth = CachedDepth(model)
        self.max_tokens = max_tokens
        self.frames_host = []
        self._compiled = None

    # ---- prompt / prefill (as stock: each branch prefilled separately, batch 1)
    def prefill(self, text):
        m = self.m
        cond = m._prompt_embeddings(text, voice=None, instruct=INSTRUCTION if self.cfg else None,
                                    ref_audio=None, ref_text=None)
        branches = [cond]
        if self.cfg:
            branches.append(m._prompt_embeddings(text, voice=None, instruct=None,
                                                 ref_audio=None, ref_text=None))
        caches, hiddens = [], []
        for emb in branches:
            c = m.backbone_model.make_cache()
            hiddens.append(m.backbone_model(input_embeddings=emb, cache=c)[:, -1, :])
            caches.append(c)
        return caches, hiddens

    # ---- one lazy frame for levels b/c/d (batched layout)
    def frame_graph(self, hidden, slot, rope_off, pad, Ks, Vs, attend_len):
        m = self.m
        logits = cfg_combine(m.lm_head(hidden), self.scale, self.batched)
        cb0 = self.sampler(logits, allow_eos=True)  # [1]
        lg, kvs = self.depth.start(hidden, cb0)
        toks = [cb0]
        for n in range(1, m.num_codebooks):
            tok = self.sampler(cfg_combine(lg, self.scale, self.batched))
            toks.append(tok)
            if n < m.num_codebooks - 1:
                lg, kvs = self.depth.step(tok, n, kvs)
        frame = mx.concatenate(toks).astype(mx.int32)  # [16]
        new_hidden, Ks, Vs = backbone_step_fn(m.backbone_model, frame, slot, rope_off, pad,
                                              Ks, Vs, attend_len, self.B)
        return cb0, frame, new_hidden, Ks, Vs

    def _get_compiled(self, cap):
        if self._compiled is None:
            def f(hidden, slot, rope_off, pad, Ks, Vs):
                return self.frame_graph(hidden, slot, rope_off, pad, Ks, Vs, cap)
            self._compiled = mx.compile(f, inputs=mx.random.state, outputs=mx.random.state)
        return self._compiled

    # ---- generation
    def run(self, text, on_audio):
        """Returns dict of timings; calls on_audio(np.ndarray) per codec chunk."""
        m = self.m
        codec = m.audio_tokenizer.decoder
        t0 = time.perf_counter()
        caches, hiddens = self.prefill(text)
        codec.reset_streaming_state()
        first_audio = [None]

        def emit(audio_arr):
            a = np.array(audio_arr.reshape(-1))  # host copy (sync)
            if first_audio[0] is None:
                first_audio[0] = time.perf_counter() - t0
            on_audio(a)

        if self.level == "a":
            eos = self._run_a(caches, hiddens, codec, emit)
        elif self.level == "b":
            eos = self._run_b(caches, hiddens, codec, emit)
        else:
            eos = self._run_cd(caches, hiddens, codec, emit)
        codec.reset_streaming_state()
        return {"wall_s": time.perf_counter() - t0, "first_audio_s": first_audio[0], "eos": eos}

    def _codec_chunk(self, codec, frames_list):
        codes = mx.array(frames_list, dtype=mx.int32)[None]
        return codec.streaming_step(mx.transpose(codes, (0, 2, 1)))

    def _run_a(self, caches, hiddens, codec, emit):
        """Stock structure (sequential CFG, .item() per sample) + depth KV cache."""
        m = self.m
        bb = m.backbone_model
        pending, eos = [], False
        for _ in range(self.max_tokens):
            logits = m.lm_head(hiddens[0])
            if self.cfg:
                u = m.lm_head(hiddens[1])
                logits = u + self.scale * (logits - u)
            first = int(self.sampler(logits, allow_eos=True).item())
            if first == m.vocab_size:
                eos = True
                break
            tok_arr = mx.array([first], dtype=mx.int32)
            states = [self.depth.start(h, tok_arr) for h in hiddens]
            frame = [first]
            for n in range(1, m.num_codebooks):
                lg = states[0][0]
                if self.cfg:
                    lu = states[1][0]
                    lg = lu + self.scale * (lg - lu)
                tok = int(self.sampler(lg).item())
                frame.append(tok)
                if n < m.num_codebooks - 1:
                    t = mx.array([tok], dtype=mx.int32)
                    states = [self.depth.step(t, n, s[1]) for s in states]
            self.frames_host.append(frame)
            pending.append(frame)
            if len(pending) >= STREAM_FRAMES:
                emit(self._codec_chunk(codec, pending[:STREAM_FRAMES]))
                del pending[:STREAM_FRAMES]
            cb = mx.array(frame, dtype=mx.int32)[None, None, :]
            hiddens = [bb(input_ids=cb, cache=c)[:, -1, :] for c in caches]
        if pending:
            emit(self._codec_chunk(codec, pending))
        return eos

    def _run_b(self, caches, hiddens, codec, emit):
        """Batched CFG (rows cond, uncond) for backbone and depth; still .item() per sample."""
        m = self.m
        buf = BackboneBuffer(m, caches, self._capacity(caches))
        hidden = mx.concatenate(hiddens, axis=0)
        pending, eos = [], False
        step = 0
        for _ in range(self.max_tokens):
            logits = cfg_combine(m.lm_head(hidden), self.scale, self.batched)
            first = int(self.sampler(logits, allow_eos=True).item())
            if first == m.vocab_size:
                eos = True
                break
            lg, kvs = self.depth.start(hidden, mx.array([first], dtype=mx.int32))
            frame = [first]
            for n in range(1, m.num_codebooks):
                tok = int(self.sampler(cfg_combine(lg, self.scale, self.batched)).item())
                frame.append(tok)
                if n < m.num_codebooks - 1:
                    lg, kvs = self.depth.step(mx.array([tok], dtype=mx.int32), n, kvs)
            self.frames_host.append(frame)
            pending.append(frame)
            if len(pending) >= STREAM_FRAMES:
                emit(self._codec_chunk(codec, pending[:STREAM_FRAMES]))
                del pending[:STREAM_FRAMES]
            slot = buf.slot + step
            hidden, buf.K, buf.V = backbone_step_fn(
                m.backbone_model, mx.array(frame, dtype=mx.int32), mx.array(slot),
                buf.lens + step, buf.pad, buf.K, buf.V, slot + 1, self.B)
            step += 1
        if pending:
            emit(self._codec_chunk(codec, pending))
        return eos

    def _capacity(self, caches):
        need = max(c[0].offset for c in caches) + self.max_tokens + 2
        return ((need + 255) // 256) * 256  # bucket so compile traces are reused across texts

    def _run_cd(self, caches, hiddens, codec, emit):
        """Lazy per-frame graph, no per-token syncs; frame n+1 is queued before frame n's
        EOS flag is read (one speculative frame). Level d compiles the frame step."""
        m = self.m
        buf = BackboneBuffer(m, caches, self._capacity(caches))
        hidden = mx.concatenate(hiddens, axis=0)
        compiled = self.level in ("d", "e")
        # level e: codec on its own GPU stream so it can overlap the depth decoder
        codec_stream = mx.new_stream(mx.gpu) if self.level == "e" else mx.default_stream(mx.gpu)
        fn = self._get_compiled(buf.cap) if compiled else None

        def build(hidden, step):
            slot = buf.slot + step
            if compiled:
                return fn(hidden, mx.array(slot), buf.lens + step, buf.pad, buf.K, buf.V)
            return self.frame_graph(hidden, mx.array(slot), buf.lens + step, buf.pad,
                                    buf.K, buf.V, slot + 1)

        eos = False
        pending = []          # lazy frame arrays awaiting codec
        audio_q = []          # lazy audio chunks awaiting host copy
        emitted_any = False
        cur = build(hidden, 0)
        cb0, frame, hidden, buf.K, buf.V = cur
        mx.async_eval(cb0, frame, hidden)
        step = 1
        for _ in range(self.max_tokens):
            # queue the next frame before blocking on this one
            nxt = None
            if step < self.max_tokens:
                nxt = build(hidden, step)
                buf.K, buf.V = nxt[3], nxt[4]
                mx.async_eval(nxt[0], nxt[1], nxt[2])
            # host check on the *current* frame (only sync per frame)
            if cb0.item() == m.vocab_size:
                eos = True
                break
            pending.append(frame)
            if len(pending) >= STREAM_FRAMES:
                codes = mx.stack(pending[:STREAM_FRAMES])[None]
                del pending[:STREAM_FRAMES]
                with mx.stream(codec_stream):
                    audio = codec.streaming_step(mx.transpose(codes, (0, 2, 1)))
                mx.async_eval(audio)
                audio_q.append(audio)
            while len(audio_q) > 1:  # copy out the previous chunk (already computed)
                emit(audio_q.pop(0))
            if audio_q and not emitted_any:
                emit(audio_q.pop(0))  # first chunk: don't wait an extra frame
                emitted_any = True
            self.frames_host.append(frame)
            if nxt is None:
                break
            cb0, frame, hidden = nxt[0], nxt[1], nxt[2]
            step += 1
        if pending:
            codes = mx.stack(pending)[None]
            with mx.stream(codec_stream):
                audio_q.append(codec.streaming_step(mx.transpose(codes, (0, 2, 1))))
        for a in audio_q:
            emit(a)
        # host frames for correctness checks (after timing-critical loop is over is fine:
        # tolist on evaluated arrays is cheap)
        self.frames_host = [f.tolist() for f in self.frames_host]
        return eos


# --------------------------------------------------------------------------- stock with frame capture
def run_stock(model, text, cfg, temperature, top_k, max_tokens, on_audio):
    frames = []
    orig = model._depth_tokens

    def rec(*a, **k):
        f = orig(*a, **k)
        frames.append(list(f))
        return f
    model._depth_tokens = rec  # instance attribute only; package untouched
    kw = {"instruct": INSTRUCTION, "cfg_scale": CFG_SCALE} if cfg else {}
    t0 = time.perf_counter()
    first = None
    try:
        for r in model.generate(text, seed=42, stream=True, streaming_interval=0.16,
                                split_pattern=None, max_tokens=max_tokens,
                                temperature=temperature, top_k=top_k, **kw):
            if first is None:
                first = time.perf_counter() - t0
            on_audio(np.array(r.audio))
    finally:
        del model._depth_tokens
    return frames, {"wall_s": time.perf_counter() - t0, "first_audio_s": first,
                    "eos": len(frames) < max_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("text")
    ap.add_argument("levels", help="comma list of stock,a,b,c,d")
    ap.add_argument("--cfg", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="top_k=1 (no repetition penalty)")
    ap.add_argument("--max", type=int, default=750)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="diagnostic: float32 activations")
    ap.add_argument("--depth-bits", type=int, default=0, help="speed probe: requantize depth decoder")
    a = ap.parse_args()
    temperature, top_p = 0.9, 1.0
    top_k = 1 if a.greedy else 50
    model = load_model(a.label)
    if a.fp32:
        make_fp32(model)
        a.tag += "_fp32"
    if a.depth_bits:
        from requant import requantize_depth
        requantize_depth(model, a.depth_bits)
        a.tag += f"_depth{a.depth_bits}bit"
    text = text_by_name(a.text)
    sr = model.sample_rate

    for level in a.levels.split(","):
        if not a.no_warm:
            # warm up kernels / compile traces on the same text (same capacity bucket)
            tw = time.perf_counter()
            if level == "stock":
                run_stock(model, text, a.cfg, temperature, top_k, 6, lambda x: None)
            else:
                mx.random.seed(1)
                Runner(model, level, a.cfg, temperature, top_p, top_k, 6).run(text, lambda x: None)
            warm_s = time.perf_counter() - tw
        else:
            warm_s = None
        chunks = []
        mx.random.seed(42)
        mx.reset_peak_memory()
        if level == "stock":
            frames, res = run_stock(model, text, a.cfg, temperature, top_k, a.max, chunks.append)
        else:
            r = Runner(model, level, a.cfg, temperature, top_p, top_k, a.max)
            res = r.run(text, chunks.append)
            frames = r.frames_host
        audio = np.concatenate(chunks) if chunks else np.zeros(0)
        audio_s = audio.shape[0] / sr
        n = len(frames)
        row = {"label": a.label, "text": a.text, "level": level, "cfg": a.cfg, "greedy": a.greedy,
               "frames": n, "eos": res["eos"], "audio_s": round(audio_s, 2),
               "wall_s": round(res["wall_s"], 2), "ms_per_frame": round(1000 * res["wall_s"] / max(n, 1), 1),
               "rtf": round(res["wall_s"] / audio_s, 3) if audio_s else None,
               "first_audio_s": round(res["first_audio_s"], 3) if res["first_audio_s"] else None,
               "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
               "warm_s": round(warm_s, 1) if warm_s is not None else None}
        stem = f"{a.label}_{a.text}_{level}_{'cfg' if a.cfg else 'nocfg'}_{'greedy' if a.greedy else 'sample'}{a.tag}"
        (PROTO / "frames").mkdir(exist_ok=True)
        (PROTO / "frames" / f"{stem}.json").write_text(json.dumps(frames))
        if a.wav:
            (PROTO / "wav").mkdir(exist_ok=True)
            write_wav(PROTO / "wav" / f"{stem}.wav", audio, sr)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
