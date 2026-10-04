"""Synthesize the 10-prompt listening set (T033, spec SC-005) against a running Breeze server.

    python3 specs/005-mlx-mac-inference/research/listening/make_set.py --url http://127.0.0.1:8080 --out <dir>

Run it unchanged against the CUDA server and against the Mac server at each precision, so every
backend gets identical requests: the same text, voice, instruction, cfg_scale and seed. Clone and
direction use the saved voices `Eric01` and `Vale01`, which exist on both machines. Writes
`<dir>/<id>.wav` and prints one line per prompt with its duration. Standard library only, so it
runs with any Python 3.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SEED = 42
# (id, kind, fields): fields are sent as the GET .wav query, the browser route.
PROMPTS = [
    ("clone-1", "clone", {"voice_id": "Eric01", "text": "I finally fixed the bug that kept the build red all week, and honestly, it was a single missing comma."}),
    ("clone-2", "clone", {"voice_id": "Vale01", "text": "Breathe in slowly, hold it for a moment, and let your shoulders drop as you breathe out."}),
    ("clone-3", "clone", {"voice_id": "Eric01", "text": "The train to the coast leaves at seven fifteen, so we should be at the station by seven."}),
    ("design-1", "design", {"instruction": "A deep, calm older male voice.", "cfg_scale": "4", "text": "Welcome back. The archive has been waiting for you, and so have I."}),
    ("design-2", "design", {"instruction": "A bright, energetic young female voice, smiling as she speaks.", "cfg_scale": "4", "text": "Good morning, everyone! Today we are going to build something amazing together."}),
    ("design-3", "design", {"instruction": "A hoarse whisper, as if telling a secret.", "cfg_scale": "4", "text": "Do not open the door at the end of the hall. Not tonight."}),
    ("direction-1", "direction", {"voice_id": "Vale01", "instruction": "Speak slowly with a restrained, serious tone.", "cfg_scale": "4", "text": "(clears throat) We need to discuss what happened last night."}),
    ("direction-2", "direction", {"voice_id": "Eric01", "instruction": "Excited and fast, barely containing laughter.", "cfg_scale": "4", "text": "You will not believe what the cat just did to the curtains."}),
    ("direction-3", "direction", {"voice_id": "Vale01", "instruction": "Sad and quiet, close to tears.", "cfg_scale": "4", "text": "I kept the letter, but I never had the courage to read it again."}),
    ("plain-1", "plain", {"text": "The quick brown fox jumps over the lazy dog, then naps in the afternoon sun."}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"{args.url}/health", timeout=10) as response:
        print(f"server: {json.load(response)}")
    failed = 0
    for prompt_id, kind, fields in PROMPTS:
        query = urllib.parse.urlencode({**fields, "seed": SEED})
        try:
            with urllib.request.urlopen(f"{args.url}/v1/audio/speech.wav?{query}", timeout=300) as response:
                body = response.read()
                version = response.headers.get("X-Breeze-Version")
        except urllib.error.HTTPError as error:
            failed += 1
            print(f"{prompt_id:12} {kind:9} FAILED {error.code} {error.read()[:200]!r}")
            continue
        (args.out / f"{prompt_id}.wav").write_bytes(body)
        # The streamed header has unknown sizes; 44 header bytes, then 16-bit mono at 24 kHz.
        seconds = (len(body) - 44) / (2 * 24000)
        print(f"{prompt_id:12} {kind:9} {seconds:6.2f} s  (server {version})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
