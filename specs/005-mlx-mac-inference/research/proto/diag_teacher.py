# ruff: noqa  (reference-only prototype; see README.md in this directory)
"""Teacher-forced bug-vs-numerics check.

Feeds the *stock* greedy frames (frames/<label>_<text>_stock_<cfg>_greedy.json) through
  ref : stock modules (backbone KVCache per branch, depth.next_logits full re-run, sequential CFG)
  new : prototype path (BackboneBuffer + backbone_step_fn over the full fixed buffer as in level d,
        CachedDepth, CFG as batch 2)
and compares the CFG-combined, reserved-masked logits at every one of the 16 sampling points
per frame. A mismatch whose reference top-2 margin is <= the observed |ref-new| error is a
numerical tie; a mismatch with a large margin would indicate a bug.

usage: diag_teacher.py <label> <text> [--cfg]
"""

import argparse
import json

import mlx.core as mx

from common import CFG_SCALE, INSTRUCTION, PROTO, load_model, make_fp32, text_by_name
from proto import BackboneBuffer, CachedDepth, backbone_step_fn

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("text")
ap.add_argument("--cfg", action="store_true")
ap.add_argument("--fp32", action="store_true")
a = ap.parse_args()

m = load_model(a.label)
if a.fp32:
    make_fp32(m)
text = text_by_name(a.text)
tag = "cfg" if a.cfg else "nocfg"
frames = json.load(open(PROTO / "frames" / f"{a.label}_{a.text}_stock_{tag}_greedy{'_fp32' if a.fp32 else ''}.json"))
S = CFG_SCALE
V = m.vocab_size
CV = m.config.codec_vocab_size


def combine(rows):
    return rows[0:1] if rows.shape[0] == 1 else rows[1:2] + S * (rows[0:1] - rows[1:2])


embs = [m._prompt_embeddings(text, voice=None, instruct=INSTRUCTION if a.cfg else None,
                             ref_audio=None, ref_text=None)]
if a.cfg:
    embs.append(m._prompt_embeddings(text, voice=None, instruct=None, ref_audio=None, ref_text=None))
caches, hs = [], []
for e in embs:
    c = m.backbone_model.make_cache()
    hs.append(m.backbone_model(input_embeddings=e, cache=c)[:, -1, :])
    caches.append(c)
# prototype buffer is built from the same prefill (as in proto.py)
need = max(c[0].offset for c in caches) + len(frames) + 2
buf = BackboneBuffer(m, caches, ((need + 255) // 256) * 256)
ref_h = hs
new_h = mx.concatenate(hs, axis=0)
depth = CachedDepth(m)

stats = {"points": 0, "argmax_mismatch": 0, "mismatch_is_tie": 0, "mismatch_not_tie": [],
         "max_abs_err": 0.0, "eos_check": None}
for fi, frame in enumerate(frames + [None]):
    # backbone head
    r = combine(mx.concatenate([m.lm_head(h) for h in ref_h], axis=0))
    n = combine(m.lm_head(new_h))
    points = [(r[0, :V + 1], n[0, :V + 1], frame[0] if frame else V, "cb0")]
    if frame is None:
        rr = r[0, :V + 1].astype(mx.float32)
        nn_ = n[0, :V + 1].astype(mx.float32)
        stats["eos_check"] = {"ref_argmax": int(mx.argmax(rr)), "new_argmax": int(mx.argmax(nn_)),
                              "eos_id": V}
        break
    # depth, teacher forced
    lg, kvs = depth.start(new_h, mx.array([frame[0]]))
    for k in range(1, 16):
        ids = mx.array([[0] + frame[:k]], dtype=mx.int32)
        rl = combine(mx.concatenate([m.depth_decoder.next_logits(ids, h) for h in ref_h], axis=0))
        points.append((rl[0, :CV], combine(lg)[0, :CV], frame[k], f"cb{k}"))
        if k < 15:
            lg, kvs = depth.step(mx.array([frame[k]]), k, kvs)
    for rl, nl, tok, name in points:
        rl, nl = rl.astype(mx.float32), nl.astype(mx.float32)
        err = float(mx.abs(rl - nl).max())
        stats["max_abs_err"] = max(stats["max_abs_err"], err)
        stats.setdefault("errs", []).append(err)
        stats["points"] += 1
        ra, na = int(mx.argmax(rl)), int(mx.argmax(nl))
        if ra != na:
            stats["argmax_mismatch"] += 1
            srt = mx.sort(rl)
            margin = float(srt[-1] - srt[-2])
            # tie: both candidates are within the observed numerical error in either path
            gap_new = float(nl[na] - nl[ra])
            stats.setdefault("mismatch_detail", []).append(
                {"frame": fi, "pt": name, "ref_margin": round(margin, 6), "new_gap": round(gap_new, 6),
                 "err": round(err, 6)})
            if margin <= 2 * err and gap_new <= 2 * err:
                stats["mismatch_is_tie"] += 1
            else:
                stats["mismatch_not_tie"].append({"frame": fi, "pt": name, "margin": margin,
                                                  "err": err, "gap_new": gap_new})
        if ra != tok:
            stats.setdefault("ref_vs_recorded_mismatch", 0)
            stats["ref_vs_recorded_mismatch"] = stats.get("ref_vs_recorded_mismatch", 0) + 1
    cb = mx.array(frame, dtype=mx.int32)
    ref_h = [m.backbone_model(input_ids=cb[None, None], cache=c)[:, -1, :] for c in caches]
    slot = buf.slot + fi
    new_h, buf.K, buf.V = backbone_step_fn(m.backbone_model, cb, mx.array(slot), buf.lens + fi,
                                           buf.pad, buf.K, buf.V, buf.cap, new_h.shape[0])
    mx.eval(new_h, ref_h, buf.K, buf.V)

import statistics
errs = stats.pop("errs")
stats["median_abs_err"] = statistics.median(errs)
md = stats.get("mismatch_detail", [])
stats["mismatch_detail"] = md[:12]
stats["mismatch_ref_margins_max"] = max((d["ref_margin"] for d in md), default=None)
stats["frames"] = len(frames)
stats["label"], stats["text"], stats["cfg"] = a.label, a.text, a.cfg
print(json.dumps(stats))
