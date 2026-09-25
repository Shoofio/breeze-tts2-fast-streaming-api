"""Client-side latency benchmark for POST /v1/audio/speech.

Drives either the current Python API (port 7860, plain form fields) or the
C++-compatible contract (port 8080, contracts/http-api.md) with the same
httpx client, so both servers can be timed the same way before and after the
API rewrite (specs/003-cpp-compatible-api/research.md R17).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

SPEECH_PATH = "/v1/audio/speech"
VOICES_PATH = "/v1/voices"
SAMPLE_RATE_FALLBACK = 24000  # only used if a response omits X-Sample-Rate
DEFAULT_PORTS = {"old": 7860, "new": 8080}
DEFAULT_REF_AUDIO = Path("$REFERENCE_VOICES_DIR/eric/eric.wav")

# short_voice needs POST /v1/voices, which the current (old) API doesn't have.
CASES_BY_API = {
    "old": ("short_design", "medium_design", "short_inline"),
    "new": ("short_design", "medium_design", "short_inline", "short_voice"),
}

INSTRUCTION = "Speak clearly and naturally."
SHORT = (
    "The quick brown fox jumps over the lazy dog, and then it takes a short "
    "nap under the old oak tree."
)
MEDIUM = (
    "Streaming speech synthesis has to balance two goals that pull in "
    "different directions. Listeners want the first sound as soon as "
    "possible, but they also want the voice to stay consistent from the "
    "first sentence to the last. A good server starts talking quickly, "
    "keeps a steady pace, and never lets the character of the voice drift "
    "halfway through a paragraph."
)


@dataclass(frozen=True)
class BenchConfig:
    api: str
    url: str
    runs: int
    ref_audio: Path
    ref_text: str | None  # None means "read <ref_audio> with a .txt suffix"
    cases: tuple[str, ...]
    voice_id: str | None


@dataclass(frozen=True)
class RunResult:
    status: int | str
    ttfa_s: float | None
    wall_s: float
    nbytes: int
    sample_rate: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", choices=("old", "new"), required=True)
    p.add_argument("--url", default=None, help="default: 7860 for old, 8080 for new")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    p.add_argument(
        "--ref-text",
        default=None,
        help="default: read the .txt file next to --ref-audio",
    )
    p.add_argument(
        "--cases",
        default=None,
        help="comma-separated; default is every case --api supports",
    )
    p.add_argument(
        "--voice-id",
        default=None,
        help="existing new-API voice_id for short_voice; registered if omitted",
    )
    return p


def parse_args(argv: list[str] | None = None) -> BenchConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    url = args.url or f"http://127.0.0.1:{DEFAULT_PORTS[args.api]}"
    cases = tuple(args.cases.split(",")) if args.cases else CASES_BY_API[args.api]

    unknown = set(cases) - set(CASES_BY_API["new"])
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    # short_voice needs POST /v1/voices, which the current (old) API lacks.
    if args.api == "old" and "short_voice" in cases:
        parser.error("short_voice is only supported for --api new")
    if args.api == "old" and args.voice_id is not None:
        parser.error("--voice-id is only supported for --api new")

    return BenchConfig(
        api=args.api,
        url=url,
        runs=args.runs,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        cases=cases,
        voice_id=args.voice_id,
    )


def post_and_drain(
    client: httpx.Client,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes, str]] | None,
) -> RunResult:
    """POST a speech request and time the first PCM byte and the full body.

    Streamed (not client.post) so `ttfa_s` reflects the first byte on the
    wire rather than the whole body. A connection that closes mid-stream
    without the chunked terminator (BC-17: failure after streaming starts)
    surfaces as `RemoteProtocolError`, recorded as a status rather than
    raised, so one bad run doesn't abort the rest of the case.
    """
    start = time.perf_counter()
    ttfa_s: float | None = None
    nbytes = 0
    status: int | str = "error:no_response"
    sample_rate = SAMPLE_RATE_FALLBACK
    try:
        with client.stream("POST", SPEECH_PATH, data=data, files=files) as resp:
            status = resp.status_code
            sample_rate = int(resp.headers.get("x-sample-rate", SAMPLE_RATE_FALLBACK))
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                if ttfa_s is None:
                    ttfa_s = time.perf_counter() - start
                nbytes += len(chunk)
    except httpx.RemoteProtocolError:
        status = "truncated"
    except httpx.HTTPError as e:
        status = f"error:{type(e).__name__}"
    wall_s = time.perf_counter() - start
    return RunResult(
        status=status, ttfa_s=ttfa_s, wall_s=wall_s, nbytes=nbytes, sample_rate=sample_rate
    )


def summarize(case: str, results: list[RunResult]) -> dict:
    """Reduce one case's runs to the medians the caller reports (R17)."""
    ok = [r for r in results if r.status == 200 and r.ttfa_s is not None]
    ttfa_ms = [r.ttfa_s * 1000 for r in ok]
    # RTF = generation wall time / audio duration; a run with no bytes has no duration.
    rtfs = [r.wall_s / (r.nbytes / 2 / r.sample_rate) for r in ok if r.nbytes]
    return {
        "event": "case_result",
        "case": case,
        "statuses": [r.status for r in results],
        "ttfa_ms_median": statistics.median(ttfa_ms) if ttfa_ms else None,
        "rtf_median": statistics.median(rtfs) if rtfs else None,
    }


def register_voice(client: httpx.Client, ref_audio: bytes, ref_text: str) -> str:
    """POST /v1/voices, left unnamed so it stays in memory and is never persisted."""
    resp = client.post(
        VOICES_PATH,
        data={"ref_text": ref_text},
        files={"ref_audio": ("ref.wav", ref_audio, "audio/wav")},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def build_cases(
    cases: tuple[str, ...], ref_audio: bytes, ref_text: str, voice_id: str | None
) -> dict[str, tuple[dict[str, str], dict | None]]:
    base = {"instruction": INSTRUCTION, "cfg_scale": "1.0", "seed": "42"}
    catalog = {
        "short_design": ({**base, "text": SHORT}, None),
        "medium_design": ({**base, "text": MEDIUM}, None),
        "short_inline": (
            {**base, "text": SHORT, "ref_text": ref_text},
            {"ref_audio": ("ref.wav", ref_audio, "audio/wav")},
        ),
        "short_voice": ({**base, "text": SHORT, "voice_id": voice_id}, None),
    }
    return {name: catalog[name] for name in cases}


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    ref_text = (
        config.ref_text
        or config.ref_audio.with_suffix(".txt").read_text().strip()
    )
    ref_audio = config.ref_audio.read_bytes()

    emit(
        {
            "event": "run_started",
            "api": config.api,
            "url": config.url,
            "runs": config.runs,
            "cases": list(config.cases),
        }
    )

    with httpx.Client(base_url=config.url, timeout=900.0) as client:
        voice_id = config.voice_id
        if "short_voice" in config.cases and voice_id is None:
            try:
                voice_id = register_voice(client, ref_audio, ref_text)
            except httpx.HTTPError as e:
                print(f"could not register the short_voice voice: {e}", file=sys.stderr)
                raise SystemExit(1) from e
            emit({"event": "voice_registered", "voice_id": voice_id})

        cases = build_cases(config.cases, ref_audio, ref_text, voice_id)
        for name, (data, files) in cases.items():
            post_and_drain(client, data, files)  # warm this shape before timing it
            results = [post_and_drain(client, data, files) for _ in range(config.runs)]
            emit(summarize(name, results))


if __name__ == "__main__":
    main()
