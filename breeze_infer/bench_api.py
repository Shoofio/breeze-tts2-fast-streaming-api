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
# HTTP never queues (contract): a 409 means "try again shortly", not "failed".
BUSY_RETRY_INTERVAL_S = 1.0
BUSY_RETRY_BUDGET_S = 60.0

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
    "halfway through a paragraph. This medium length passage exists to "
    "measure exactly that trade-off."
)


@dataclass(frozen=True)
class BenchConfig:
    api: str
    url: str
    runs: int
    ref_audio: Path | None
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
    p.add_argument("--api", choices=("old", "new"), default="new")
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


def _parse_cases(parser: argparse.ArgumentParser, raw: str | None, api: str) -> tuple[str, ...]:
    if raw is None:
        return CASES_BY_API[api]
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if len(set(names)) != len(names):
        parser.error("--cases has a duplicate case name")
    unknown = set(names) - set(CASES_BY_API["new"])
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    return tuple(names)


def parse_args(argv: list[str] | None = None) -> BenchConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    url = args.url or f"http://127.0.0.1:{DEFAULT_PORTS[args.api]}"
    cases = _parse_cases(parser, args.cases, args.api)

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


def needs_reference(cases: tuple[str, ...], voice_id: str | None) -> bool:
    """short_inline always needs the inline reference; short_voice only if it
    still has to register one (an explicit --voice-id already has a voice)."""
    return "short_inline" in cases or ("short_voice" in cases and voice_id is None)


def load_reference(ref_audio_path: Path, ref_text: str | None) -> tuple[bytes, str]:
    """Read the inline reference lazily, only when a selected case needs it.

    An explicit `--ref-text ""` is sent as given (the server then rejects
    it), so only `ref_text is None` falls back to the sibling .txt file. A
    missing file exits cleanly instead of raising, since argument problems
    shouldn't show a traceback.
    """
    try:
        ref_audio = ref_audio_path.read_bytes()
        text = (
            ref_text
            if ref_text is not None
            else ref_audio_path.with_suffix(".txt").read_text().strip()
        )
    except OSError as e:
        raise SystemExit(f"could not read the reference audio/text: {e}") from e
    return ref_audio, text


def post_and_drain(
    client: httpx.Client,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes, str]] | None,
) -> RunResult:
    """POST a speech request and time the first PCM byte and the full body.

    Streamed (not client.post) so `ttfa_s` reflects the first byte on the
    wire rather than the whole body. Distinguishes two failure shapes:
    - the connection dies before any status line arrives (`error:<type>`);
    - it dies mid-stream, after a status was already read (`truncated`,
      BC-17: failure after streaming starts closes the connection without
      the chunked terminator).
    Either way this returns a status rather than raising, so one bad run
    doesn't abort the rest of the case.
    """
    start = time.perf_counter()
    ttfa_s: float | None = None
    nbytes = 0
    status: int | str = "error:no_response"
    status_received = False
    sample_rate = SAMPLE_RATE_FALLBACK
    try:
        with client.stream("POST", SPEECH_PATH, data=data, files=files) as resp:
            status = resp.status_code
            status_received = True
            try:
                sample_rate = int(resp.headers.get("x-sample-rate", SAMPLE_RATE_FALLBACK))
            except ValueError:
                status = "error:bad_sample_rate"
            else:
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    if ttfa_s is None:
                        ttfa_s = time.perf_counter() - start
                    nbytes += len(chunk)
    except httpx.RemoteProtocolError:
        status = "truncated" if status_received else "error:RemoteProtocolError"
    except httpx.HTTPError as e:
        status = f"error:{type(e).__name__}"
    wall_s = time.perf_counter() - start
    return RunResult(
        status=status, ttfa_s=ttfa_s, wall_s=wall_s, nbytes=nbytes, sample_rate=sample_rate
    )


def post_with_retry(
    client: httpx.Client,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes, str]] | None,
    *,
    sleep=time.sleep,
    now=time.perf_counter,
) -> RunResult:
    """Poll through a 409 (busy) instead of counting a momentary clash as a
    failed run. Bounded at `BUSY_RETRY_BUDGET_S`; a run still busy when the
    budget runs out is reported as "busy" rather than retried forever.
    `sleep`/`now` are injected so tests never wait for real time to pass.
    """
    deadline = now() + BUSY_RETRY_BUDGET_S
    result = post_and_drain(client, data, files)
    while result.status == 409 and now() < deadline:
        sleep(BUSY_RETRY_INTERVAL_S)
        result = post_and_drain(client, data, files)
    if result.status == 409:
        return RunResult(
            status="busy", ttfa_s=None, wall_s=result.wall_s, nbytes=0,
            sample_rate=result.sample_rate,
        )
    return result


def summarize(case: str, results: list[RunResult]) -> dict:
    """Reduce one case's runs to the medians the caller reports (R17), plus
    the per-run values behind them."""
    ok = [r for r in results if r.status == 200 and r.ttfa_s is not None]
    ttfa_ms = [r.ttfa_s * 1000 for r in ok]
    # RTF = generation wall time / audio duration; a run with no bytes has no duration.
    ok_with_audio = [r for r in ok if r.nbytes]
    audio_s = [r.nbytes / 2 / r.sample_rate for r in ok_with_audio]
    rtfs = [r.wall_s / seconds for r, seconds in zip(ok_with_audio, audio_s)]
    return {
        "event": "case_result",
        "case": case,
        "statuses": [r.status for r in results],
        "ttfa_ms": ttfa_ms,
        "rtf": rtfs,
        "ttfa_ms_median": statistics.median(ttfa_ms) if ttfa_ms else None,
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "audio_s_median": statistics.median(audio_s) if audio_s else None,
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


def delete_voice(client: httpx.Client, voice_id: str) -> None:
    """Best-effort cleanup for a voice this run registered itself (never one
    supplied via --voice-id). A failed delete is reported, not raised: it
    shouldn't take down a benchmark run that otherwise succeeded."""
    try:
        client.delete(f"{VOICES_PATH}/{voice_id}").raise_for_status()
    except httpx.HTTPError as e:
        print(f"could not delete voice {voice_id}: {e}", file=sys.stderr)


def build_cases(
    cases: tuple[str, ...],
    ref_audio: bytes | None,
    ref_text: str | None,
    voice_id: str | None,
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


def run_benchmark(client: httpx.Client, config: BenchConfig, *, sleep=time.sleep) -> None:
    """Run every selected case against `client` and emit each as a JSON event."""
    voice_id = config.voice_id
    ref_audio = ref_text = None
    if needs_reference(config.cases, voice_id):
        ref_audio, ref_text = load_reference(config.ref_audio, config.ref_text)

    registered_voice = False
    if "short_voice" in config.cases and voice_id is None:
        try:
            voice_id = register_voice(client, ref_audio, ref_text)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            print(f"could not register the short_voice voice: {e}", file=sys.stderr)
            raise SystemExit(1) from e
        registered_voice = True
        emit({"event": "voice_registered", "voice_id": voice_id})

    try:
        cases = build_cases(config.cases, ref_audio, ref_text, voice_id)
        for name, (data, files) in cases.items():
            warmup = post_with_retry(client, data, files, sleep=sleep)  # warm this shape first
            if warmup.status != 200:
                emit({"event": "warmup_failed", "case": name, "status": warmup.status})
            results = [post_with_retry(client, data, files, sleep=sleep) for _ in range(config.runs)]
            emit(summarize(name, results))
    finally:
        if registered_voice:
            delete_voice(client, voice_id)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
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
        run_benchmark(client, config)


if __name__ == "__main__":
    main()
