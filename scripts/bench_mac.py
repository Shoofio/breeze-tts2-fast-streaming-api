"""Streaming latency on a running server, the way a voice assistant uses it: one line at a time, streamed.

For each line: first audio (request to first PCM byte), real-time factor, and "hold", the buffer playback needs so it
never runs dry (0 = every chunk arrived before it was due). First audio + hold is when playback can start without a
stall, the number that says whether a backend is real time.

    python scripts/bench_mac.py --ref-audio REF.wav --ref-text "its transcript" [--url http://127.0.0.1:8080]
        [--cfg 2] [--repeat 1] [--out DIR] [--label NAME]

The lines are Harvard sentences (IEEE 1969, public domain) of one to four sentences; two in three carry a voice
direction at --cfg (the rest are a plain clone at CFG 1). The reference is registered once with POST /v1/voices, so
its prefix stays cached, as a long-running assistant would use it. --out keeps each line's WAV and results.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import httpx

RATE = 24000
HARVARD = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
]
DIRECTIONS = [
    "Speak warmly and gently, with a soft smile in the voice.",
    "Speak playfully, with a teasing, amused lilt.",
    "Speak with bright, excited energy, a little faster than usual.",
    "Speak slowly and calmly, in a soothing, reassuring tone.",
    "Speak softly, with gentle sympathy and concern.",
    "Speak in a restrained, serious, matter-of-fact tone.",
    "Speak in a dry, deadpan tone, understated.",
    "Whisper softly and intimately.",
]


def lines() -> list[dict]:
    """24 lines: lengths 1, 2 and 4 sentences in turn; every third plain, the rest directed."""
    out, k = [], 0
    for i in range(24):
        n = (1, 2, 4)[i % 3]
        text = " ".join(HARVARD[(k + j) % len(HARVARD)] for j in range(n))
        k += n
        out.append({"text": text, "instruction": None if i % 3 == 0 else DIRECTIONS[i % len(DIRECTIONS)]})
    return out


def register(client: httpx.Client, url: str, ref_audio: Path, ref_text: str) -> str:
    with ref_audio.open("rb") as f:
        r = client.post(f"{url}/v1/voices", files={"ref_audio": (ref_audio.name, f, "audio/wav")}, data={"ref_text": ref_text})
    r.raise_for_status()
    return r.json()["id"]


def synthesize(client: httpx.Client, url: str, voice: str, line: dict, cfg: float) -> tuple[dict, bytes]:
    data = {"text": line["text"], "voice_id": voice, "seed": "42", "cfg_scale": "1"}
    if line.get("instruction"):
        data.update(instruction=line["instruction"], cfg_scale=str(cfg))
    arrivals, pcm = [], bytearray()  # (seconds since the request, audio seconds received by then)
    t0 = time.perf_counter()
    while True:
        with client.stream("POST", f"{url}/v1/audio/speech", data=data) as r:
            if r.status_code == 409:  # the previous stream is still being torn down
                time.sleep(0.01)
                continue
            r.raise_for_status()
            for chunk in r.iter_bytes():
                if chunk:
                    pcm += chunk
                    arrivals.append((time.perf_counter() - t0, len(pcm) / 2 / RATE))
        break
    total = time.perf_counter() - t0
    first, first_audio = arrivals[0]
    # Playback starts at first + hold; each chunk must arrive before playback reaches the audio received before it.
    hold, before = 0.0, 0.0
    for t, received in arrivals:
        hold = max(hold, t - first - before)
        before = received
    audio = len(pcm) / 2 / RATE
    return {
        "first": first,
        "hold": hold,
        "total": total,
        "audio": audio,
        "rtf": (total - first) / max(audio - first_audio, 1e-6),
        "directed": bool(line.get("instruction")),
    }, bytes(pcm)


def summary(rows: list[dict], key: str) -> tuple[float, float]:
    values = sorted(r[key] for r in rows)
    return statistics.median(values), values[min(len(values) - 1, round(0.9 * (len(values) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    client = httpx.Client(timeout=120)
    voice = register(client, a.url, a.ref_audio, a.ref_text)
    for warm in ({"text": "Warm up, one two three."}, {"text": "Warm up, one two three.", "instruction": DIRECTIONS[0]}):
        synthesize(client, a.url, voice, warm, a.cfg)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(a.repeat):
        for i, line in enumerate(lines()):
            row, pcm = synthesize(client, a.url, voice, line, a.cfg)
            rows.append(row)
            print(f"{i:2d} {'directed' if row['directed'] else 'plain':8s} first {row['first'] * 1e3:4.0f} ms  "
                  f"hold {row['hold'] * 1e3:4.0f} ms  rtf {row['rtf']:.2f}  {row['audio']:4.1f} s", flush=True)
            if a.out and rep == 0:
                with wave.open(str(a.out / f"{i:02d}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(pcm)
    print(f"\n{a.label}")
    for name, directed in (("plain", False), ("directed", True)):
        rs = [r for r in rows if r["directed"] == directed]
        start = summary([{"s": r["first"] + r["hold"]} for r in rs], "s")
        first, rtf = summary(rs, "first"), summary(rs, "rtf")
        stalls = sum(r["hold"] > 0.05 for r in rs)
        print(f"{name:8s} n={len(rs):2d}  first p50 {first[0] * 1e3:4.0f} ms | stall-free start p50 {start[0] * 1e3:4.0f} "
              f"p90 {start[1] * 1e3:4.0f} ms | rtf p50 {rtf[0]:.2f} p90 {rtf[1]:.2f} | lines that stalled {stalls}")
    if a.out:
        (a.out / "results.json").write_text(json.dumps({"label": a.label, "cfg": a.cfg, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
