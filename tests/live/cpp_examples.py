"""Run the C++ server's documented HTTP and WebSocket examples against this server (SC-001, T080).

    python -m tests.live.cpp_examples --url http://127.0.0.1:8080 [--cors [ORIGINS]] [--only 3,20]

Each example from `Breeze-TTS-2.cpp/docs/server.md`, `voices.md` and `websocket.md` is sent to
the server, and the status, the documented headers and the body or event shape are compared with
what the C++ docs say. Every difference must match an entry in `EXPECTED_DIFFERENCES`, keyed by
the spec's Breaking Change id (spec.md "Breaking Changes from the C++ Server"), or by an `ADD-n`
id for the spec's "Additive changes (not breaking)" paragraph. Any other difference, or an example
that fails to run, makes the exit status 1.

The examples are transcribed as data and code, one function per example with the doc section
named in its decorator, rather than parsed out of the markdown. The docs' code blocks aren't
regular enough to parse: the request is a curl command, a shell pipe or a Python sketch, the
response sits in a separate block, table or sentence, and the WebSocket session is a `->`/`<-`
transcript. The doc's host and port (`127.0.0.1:8137`, `8081`) become `--url` and the `ws_port`
that `GET /health` reports.

Voice safety: the harness never registers or deletes `eric` or `vale`. A voice an example
registers is named `st_live_tmp_cppNN` (NN is the example number) or is an unnamed `v_` id this
run created, and is deleted afterwards, even when the example fails. Leftover `st_live_tmp_cpp*`
voices from an interrupted run are swept before and after the run. The reference clip is only
upload content (by default the `eric` source recording, used under the throwaway names).

`EXPECTED_DIFFERENCES` was derived from contracts/http-api.md and contracts/ws-api.md before any
live run; entries marked `verify_live` are to be confirmed in the T086 live gate.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fnmatch
import inspect
import json
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from websockets.asyncio.client import ClientConnection, connect

C_DOCS = Path("<Breeze-TTS-2.cpp checkout>/docs")
DEFAULT_REF_WAV = Path("$REFERENCE_VOICES_DIR/eric/eric.wav")

# Every voice this harness names starts with this prefix; the sweep deletes only these.
TMP_PREFIX = "st_live_tmp_cpp"
# Voices that belong to the user. Checked explicitly as a second guard next to the prefix rule.
PROTECTED_VOICES = frozenset({"eric", "vale"})

HTTP_TIMEOUT = 120.0
WS_STEP_TIMEOUT = 60.0
# The C++ docs tell HTTP clients to retry a 409 with backoff; examples other than the busy one
# do that, so a generation still winding down from the previous example doesn't fail the next.
BUSY_RETRIES = 30
BUSY_BACKOFF = 0.5


# --------------------------------------------------------------------------------------------
# Comparison model (pure; unit-tested in tests/test_cpp_examples.py)
# --------------------------------------------------------------------------------------------


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# A key or header that isn't there. Distinct from None, which is JSON null.
MISSING = _Sentinel("<absent>")
# In a Match: any value fits.
ANY = _Sentinel("<any>")


@dataclass(frozen=True)
class Pred:
    """An expected value described by a rule rather than a literal (a shape, not a value)."""

    description: str
    test: Callable[[object], bool]

    def __repr__(self) -> str:
        return f"<{self.description}>"


STRING = Pred("a string", lambda v: isinstance(v, str))
INT = Pred("an integer", lambda v: isinstance(v, int) and not isinstance(v, bool))
POSITIVE_INT = Pred("an integer above 0", lambda v: INT.test(v) and v > 0)
NUMBER = Pred("a number", lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
BOOL = Pred("a boolean", lambda v: isinstance(v, bool))


def compare(where: str, cpp: object, actual: object) -> list[tuple[str, object, object]]:
    """Compare what the C++ docs say (`cpp`) with what this server did (`actual`).

    Returns `(where, cpp, actual)` for each difference; `where` is a dotted path such as
    `body.file_kept` or `events.2.text`. Dicts are compared key by key, so an extra key and a
    missing key are both differences (the doc bodies are exact). Lists of the same length are
    compared element by element; a length mismatch is one difference for the whole list.
    """
    if isinstance(cpp, Pred):
        return [] if actual is not MISSING and cpp.test(actual) else [(where, cpp, actual)]
    if isinstance(cpp, dict) and isinstance(actual, dict):
        diffs: list[tuple[str, object, object]] = []
        for key, value in cpp.items():
            diffs += compare(f"{where}.{key}", value, actual.get(key, MISSING))
        diffs += [(f"{where}.{key}", MISSING, value) for key, value in actual.items() if key not in cpp]
        return diffs
    if isinstance(cpp, list) and isinstance(actual, list) and len(cpp) == len(actual):
        diffs = []
        for index, (c, a) in enumerate(zip(cpp, actual, strict=True)):
            diffs += compare(f"{where}.{index}", c, a)
        return diffs
    # Type too, so True never passes for 1 and 24000.0 never passes for 24000.
    if type(cpp) is type(actual) and cpp == actual:
        return []
    return [(where, cpp, actual)]


@dataclass(frozen=True)
class Difference:
    example: str  # example id, e.g. "cpp18"
    where: str
    cpp: object
    actual: object


@dataclass(frozen=True)
class Match:
    """One difference an expected-difference entry explains.

    `example` and `where` are fnmatch patterns over the example id and the dotted path. `cpp`
    and `actual` are a literal, a `Pred`, `MISSING` or `ANY`.
    """

    example: str
    where: str
    cpp: object = ANY
    actual: object = ANY


@dataclass(frozen=True)
class ExpectedDifference:
    summary: str
    matches: tuple[Match, ...]
    # Derived from the contracts, not yet seen against a live server (T086 confirms).
    verify_live: bool = False


def _fits(pattern: object, value: object) -> bool:
    if pattern is ANY:
        return True
    if isinstance(pattern, Pred):
        return value is not MISSING and pattern.test(value)
    return type(pattern) is type(value) and pattern == value


def _matches(match: Match, diff: Difference) -> bool:
    return (
        fnmatch.fnmatchcase(diff.example, match.example)
        and fnmatch.fnmatchcase(diff.where, match.where)
        and _fits(match.cpp, diff.cpp)
        and _fits(match.actual, diff.actual)
    )


# Every difference from the C++ docs this server is allowed to show. Keys are spec.md ids.
EXPECTED_DIFFERENCES: dict[str, ExpectedDifference] = {
    "BC-28": ExpectedDifference(
        summary="DELETE removes a saved voice's file, so file_kept is always false",
        matches=(Match("cpp18", "body.file_kept", cpp=True, actual=False),),
        verify_live=True,
    ),
    # spec.md BC-18: "`OPTIONS` without CORS gets `404`" in C++, "`405` with `Allow`" here;
    # contracts/http-api.md CORS section: "every `OPTIONS` request gets `405 method_not_allowed`".
    "BC-18": ExpectedDifference(
        summary="OPTIONS without CORS is 405 with Allow, not the C++ server's 404",
        matches=(Match("cpp12", "status", cpp=404, actual=405),),
        verify_live=True,
    ),
    # spec.md "Additive changes (not breaking)": "an error `code` field alongside `error` on
    # HTTP and WebSocket errors" (http-api.md Errors, ws-api.md error event).
    "ADD-2": ExpectedDifference(
        summary="Additive: every error carries a machine-readable `code` next to its message",
        matches=(
            Match("*", "body.code", cpp=MISSING, actual=STRING),
            Match("*", "*events.*.code", cpp=MISSING, actual=STRING),
        ),
        verify_live=True,
    ),
    # spec.md "Additive changes": "`type` on WebSocket error events"; ws-api.md names the field
    # `request_type`: the client message type that caused the error, or null.
    "ADD-3": ExpectedDifference(
        summary="Additive: WebSocket error events carry `request_type`, the message that caused them",
        matches=(
            Match("*", "*events.*.request_type", cpp=MISSING, actual=Pred("a string or null", lambda v: v is None or isinstance(v, str))),
        ),
        verify_live=True,
    ),
}


def explain(diff: Difference, table: Mapping[str, ExpectedDifference] = EXPECTED_DIFFERENCES) -> str | None:
    """The id of the first entry that explains `diff`, or None when nothing does."""
    for key, entry in table.items():
        if any(_matches(match, diff) for match in entry.matches):
            return key
    return None


@dataclass
class ExampleResult:
    example_id: str
    title: str
    diffs: list[Difference] = field(default_factory=list)
    error: str | None = None  # the example could not run to the end
    skipped: str | None = None  # the example needs a launch option the server wasn't given
    notes: list[str] = field(default_factory=list)


def outcome(result: ExampleResult, table: Mapping[str, ExpectedDifference] = EXPECTED_DIFFERENCES) -> str:
    """PASS, EXPLAINED (every difference has an entry), FAIL, or SKIP."""
    if result.skipped is not None:
        return "SKIP"
    if result.error is not None or any(explain(d, table) is None for d in result.diffs):
        return "FAIL"
    return "EXPLAINED" if result.diffs else "PASS"


def exit_code(results: Sequence[ExampleResult], table: Mapping[str, ExpectedDifference] = EXPECTED_DIFFERENCES) -> int:
    """1 when any example has an unexplained difference or failed to run, else 0."""
    return 1 if any(outcome(r, table) == "FAIL" for r in results) else 0


def render_report(results: Sequence[ExampleResult], table: Mapping[str, ExpectedDifference] = EXPECTED_DIFFERENCES) -> list[str]:
    """The per-example report lines plus a summary."""
    lines: list[str] = []
    counts = {"PASS": 0, "EXPLAINED": 0, "FAIL": 0, "SKIP": 0}
    seen: set[str] = set()
    for result in results:
        verdict = outcome(result, table)
        counts[verdict] += 1
        lines.append(f"{verdict:<9} {result.example_id}  {result.title}")
        if result.skipped is not None:
            lines.append(f"            skipped: {result.skipped}")
        for note in result.notes:
            lines.append(f"            note: {note}")
        for diff in result.diffs:
            key = explain(diff, table)
            head = f"            {diff.where}: C++ {diff.cpp!r}, here {_short(diff.actual)}"
            if key is None:
                lines.append(f"{head}  -> UNEXPLAINED")
            else:
                seen.add(key)
                live = ", verify live" if table[key].verify_live else ""
                lines.append(f"{head}  -> {key}{live}: {table[key].summary}")
        if result.error is not None:
            lines.append(f"            error: {result.error}")
    lines.append("")
    lines.append(
        f"{len(results)} examples: {counts['PASS']} pass, {counts['EXPLAINED']} explained, "
        f"{counts['FAIL']} fail, {counts['SKIP']} skipped"
    )
    unused = [key for key in table if key not in seen]
    if unused:
        lines.append(f"Expected differences not seen this run: {', '.join(unused)}")
    live = [key for key in table if table[key].verify_live and key in seen]
    if live:
        lines.append(f"Seen entries still marked verify live (confirm, then clear the flag): {', '.join(live)}")
    lines.append("RESULT: " + ("FAIL (unexplained differences above)" if exit_code(results, table) else "OK"))
    return lines


def _short(value: object, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def require_throwaway(voice_id: str, owned: frozenset[str] | set[str] = frozenset()) -> None:
    """Refuse to touch any voice but a throwaway one: `st_live_tmp_cpp*` or an id this run made."""
    if voice_id.lower() in PROTECTED_VOICES:
        raise ValueError(f"refusing to touch the protected voice {voice_id!r}")
    if not (voice_id.startswith(TMP_PREFIX) or voice_id in owned):
        raise ValueError(f"refusing to touch {voice_id!r}: not a throwaway voice of this run")


# --------------------------------------------------------------------------------------------
# Running against a server
# --------------------------------------------------------------------------------------------


@dataclass
class Context:
    http: httpx.Client
    ref_wav: bytes
    ref_text: str
    cors: str | None  # None: launched without --cors; "*": --cors; else the allowlist
    run_tag: str
    sample_rate: int = 0
    ws_url: str = ""
    owned: set[str] = field(default_factory=set)  # unnamed voice ids this run registered


class Run:
    """Collects one example's differences."""

    def __init__(self, example_id: str, number: int) -> None:
        self.example_id = example_id
        self.tmp_name = f"{TMP_PREFIX}{number:02d}"
        self.diffs: list[Difference] = []
        self.notes: list[str] = []

    def check(self, where: str, cpp: object, actual: object) -> None:
        self.diffs += [Difference(self.example_id, w, c, a) for w, c, a in compare(where, cpp, actual)]


class Skip(Exception):
    """The example needs a server launch option this run doesn't have."""


@dataclass(frozen=True)
class Example:
    number: int
    doc: str
    section: str
    fn: Callable[[Context, Run], object]

    @property
    def example_id(self) -> str:
        return f"cpp{self.number:02d}"

    @property
    def title(self) -> str:
        return f"{self.doc} > {self.section}"


EXAMPLES: list[Example] = []


def example(number: int, doc: str, section: str) -> Callable:
    """Register an example. Numbers are fixed so ids and throwaway names stay stable."""

    def register(fn: Callable[[Context, Run], object]) -> Callable[[Context, Run], object]:
        if any(e.number == number for e in EXAMPLES):
            raise ValueError(f"example number {number} used twice")
        EXAMPLES.append(Example(number, doc, section, fn))
        return fn

    return register


def form(fields: Mapping[str, str], ref_audio: bytes | None = None) -> list[tuple[str, tuple]]:
    """A multipart body, as `curl --form-string` / `-F ref_audio=@reference.wav` sends it."""
    parts: list[tuple[str, tuple]] = [(name, (None, value)) for name, value in fields.items()]
    if ref_audio is not None:
        parts.append(("ref_audio", ("reference.wav", ref_audio, "audio/wav")))
    return parts


def retry_busy(send: Callable[[], httpx.Response]) -> httpx.Response:
    response = send()
    for _ in range(BUSY_RETRIES):
        if response.status_code != 409:
            break
        time.sleep(BUSY_BACKOFF)
        response = send()
    return response


def post_speech(ctx: Context, fields: Mapping[str, str], ref_audio: bytes | None = None) -> httpx.Response:
    return retry_busy(lambda: ctx.http.post("/v1/audio/speech", files=form(fields, ref_audio)))


def body_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return f"<not JSON: {response.text[:120]!r}>"


def media_type(response: httpx.Response) -> object:
    value = response.headers.get("content-type")
    return MISSING if value is None else value.split(";")[0].strip()


def header(response: httpx.Response, name: str) -> object:
    value = response.headers.get(name)
    return MISSING if value is None else value


PCM_BODY = Pred("non-empty headerless s16le PCM (even length, not a RIFF file)",
                lambda b: isinstance(b, int) and b > 0 and b % 2 == 0)


def check_pcm(run: Run, response: httpx.Response, prefix: str = "") -> None:
    """The `200` response table in server.md "POST /v1/audio/speech > Response"."""
    run.check(f"{prefix}status", 200, response.status_code)
    if response.status_code != 200:
        run.check(f"{prefix}body", "PCM audio", body_json(response))
        return
    run.check(f"{prefix}header.content-type", "audio/pcm", media_type(response))
    run.check(f"{prefix}header.transfer-encoding", "chunked", header(response, "transfer-encoding"))
    run.check(f"{prefix}header.x-sample-rate", "24000", header(response, "x-sample-rate"))
    run.check(f"{prefix}header.x-sample-format", "s16le", header(response, "x-sample-format"))
    run.check(f"{prefix}header.cache-control", "no-store", header(response, "cache-control"))
    body = response.content
    run.check(f"{prefix}body.bytes", PCM_BODY, -1 if body.startswith(b"RIFF") else len(body))


def voice_shape(voice_id: object, saved: bool, ref_text: object) -> dict:
    """The voice object in voices.md "Listing and removing"."""
    return {"id": voice_id, "frames": INT, "seconds": NUMBER, "encode_ms": INT, "saved": saved, "ref_text": ref_text}


def register_voice(ctx: Context, name: str | None, ref_text: str) -> httpx.Response:
    if name is not None:
        require_throwaway(name)
    fields = {"ref_text": ref_text} if name is None else {"ref_text": ref_text, "name": name}
    response = retry_busy(lambda: ctx.http.post("/v1/voices", files=form(fields, ctx.ref_wav)))
    if name is None and response.status_code == 200:
        ctx.owned.add(response.json()["id"])
    return response


def delete_voice(ctx: Context, voice_id: str) -> httpx.Response:
    require_throwaway(voice_id, ctx.owned)
    return ctx.http.delete(f"/v1/voices/{voice_id}")


@contextlib.contextmanager
def temp_voice(ctx: Context, run: Run) -> Iterator[str]:
    """A saved voice named `st_live_tmp_cppNN`, deleted afterwards even if the example fails."""
    response = register_voice(ctx, run.tmp_name, ctx.ref_text)
    if response.status_code != 200:
        raise RuntimeError(f"could not register {run.tmp_name}: {response.status_code} {response.text[:200]}")
    try:
        yield run.tmp_name
    finally:
        delete_voice(ctx, run.tmp_name)


def sweep(ctx: Context) -> list[str]:
    """Delete leftover `st_live_tmp_cpp*` voices and this run's unnamed ones."""
    response = ctx.http.get("/v1/voices")
    if response.status_code != 200:
        # Say so: a silent skip would leave throwaway voices behind with no sign of it.
        print(f"WARNING: sweep could not list voices (GET /v1/voices gave {response.status_code}); "
              f"delete {TMP_PREFIX}* voices by hand", file=sys.stderr)
        return []
    ids = [v["id"] for v in response.json() if v["id"].startswith(TMP_PREFIX) or v["id"] in ctx.owned]
    for voice_id in ids:
        status = delete_voice(ctx, voice_id).status_code
        # 404 is fine: an example's own cleanup already removed it.
        if status not in (200, 404):
            print(f"WARNING: sweep could not delete {voice_id} (DELETE gave {status}); delete it by hand",
                  file=sys.stderr)
    return ids


# ---- WebSocket helpers --------------------------------------------------------------------

AUDIO = "<audio>"  # a binary frame, in a probe's item list


class WsProbe:
    """Records everything a socket receives: JSON events as dicts, binary frames as AUDIO."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.items: list[object] = []
        self.audio_bytes = 0

    async def send(self, **message: object) -> None:
        await self.ws.send(json.dumps(message))

    async def _receive(self, timeout: float) -> object:
        message = await asyncio.wait_for(self.ws.recv(), timeout)
        if isinstance(message, bytes):
            self.audio_bytes += len(message)
            item: object = AUDIO
        else:
            item = json.loads(message)
        self.items.append(item)
        return item

    async def until(self, *kinds: str, timeout: float = WS_STEP_TIMEOUT) -> object:
        """Read until an event whose type is in `kinds` (or AUDIO) arrives; return it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            try:
                item = await self._receive(max(0.0, deadline - loop.time()))
            except TimeoutError:
                raise TimeoutError(f"no {'/'.join(kinds)} within {timeout:.0f} s; received {self.summary()}") from None
            if item == AUDIO and AUDIO in kinds:
                return item
            if isinstance(item, dict) and item.get("type") in kinds:
                return item

    async def collect_for(self, seconds: float) -> None:
        """Read whatever arrives in the next `seconds`."""
        with contextlib.suppress(TimeoutError):
            loop = asyncio.get_running_loop()
            deadline = loop.time() + seconds
            while (left := deadline - loop.time()) > 0:
                await self._receive(left)

    def summary(self) -> str:
        return repr(collapse_audio(self.items))


def collapse_audio(items: Sequence[object]) -> list[object]:
    """Items with each run of binary frames shown once, for readable error messages."""
    out: list[object] = []
    for item in items:
        if not (item == AUDIO and out and out[-1] == AUDIO):
            out.append(item)
    return out


def json_events(items: Sequence[object], drop: tuple[str, ...] = ("ready", "queued")) -> list[dict]:
    """The JSON events in order. `queued` is dropped unless an example is about it, because it
    only says the GPU was busy at that moment, which the docs' transcripts don't show."""
    return [i for i in items if isinstance(i, dict) and i.get("type") not in drop]


def audio_after_each_speaking(items: Sequence[object]) -> bool:
    """Each `speaking` is followed by at least one binary frame before the piece's end."""
    waiting = False
    for item in items:
        if item == AUDIO:
            waiting = False
        elif isinstance(item, dict) and item.get("type") in ("speaking", "done", "cancelled"):
            if waiting:
                return False
            waiting = item["type"] == "speaking"
    return not waiting


@contextlib.asynccontextmanager
async def ws_open(ctx: Context):
    if not ctx.ws_url:
        raise Skip("GET /health reported ws_port 0, so the WebSocket is disabled or failed to bind")
    # proxy=None: a local server must not be reached through an environment proxy.
    async with connect(ctx.ws_url, max_size=None, open_timeout=10, proxy=None) as ws:
        yield WsProbe(ws)


def check_ready(ctx: Context, run: Run, ready: object) -> None:
    run.check("ready", {"type": "ready", "sample_rate": ctx.sample_rate, "format": "s16le"}, ready)


# --------------------------------------------------------------------------------------------
# server.md
# --------------------------------------------------------------------------------------------

HEALTH_BODY = {"status": "ok", "sample_rate": 24000, "ws_port": POSITIVE_INT}


@example(1, "server.md", "GET /health: curl http://127.0.0.1:8137/health")
def health(ctx: Context, run: Run) -> None:
    response = ctx.http.get("/health")
    run.check("status", 200, response.status_code)
    run.check("body", HEALTH_BODY, body_json(response))


@example(2, "server.md", "Streaming without stutter > Real time factor: curl -w time_total")
def real_time_factor(ctx: Context, run: Run) -> None:
    started = time.monotonic()
    response = post_speech(ctx, {"text": "The tide came in slowly over the rocks while the gulls circled overhead."})
    elapsed = time.monotonic() - started
    check_pcm(run, response)
    if response.status_code == 200 and elapsed > 0:
        seconds = len(response.content) / 2 / 24000
        run.notes.append(f"{seconds:.2f} s of audio in {elapsed:.2f} s, real time factor {seconds / elapsed:.2f}x")


@example(3, "server.md", "POST /v1/audio/speech > Examples: voice design")
def voice_design(ctx: Context, run: Run) -> None:
    response = post_speech(ctx, {
        "text": "Welcome aboard. Your journey begins now.",
        "instruction": "A warm, thoughtful young woman with a clear, calm delivery.",
        "cfg_scale": "1",
        "seed": "42",
    })
    check_pcm(run, response)


@example(4, "server.md", "POST /v1/audio/speech > Examples: voice clone")
def voice_clone(ctx: Context, run: Run) -> None:
    # The doc's placeholder transcript is replaced by the real one for the uploaded clip.
    response = post_speech(ctx, {"text": "It is good to hear your voice again.", "ref_text": ctx.ref_text}, ctx.ref_wav)
    check_pcm(run, response)


@example(5, "server.md", "POST /v1/audio/speech > Examples: voice direction")
def voice_direction(ctx: Context, run: Run) -> None:
    response = post_speech(ctx, {
        "text": "We need to discuss what happened last night.",
        "instruction": "Speak slowly with a restrained, serious tone.",
        "ref_text": ctx.ref_text,
        "cfg_scale": "1",
    }, ctx.ref_wav)
    check_pcm(run, response)


@example(6, "server.md", "POST /v1/audio/speech > Examples: play the raw stream (curl -sN | ffplay)")
def raw_stream(ctx: Context, run: Run) -> None:
    response = post_speech(ctx, {"text": "Hello there."})
    check_pcm(run, response)


@example(7, "server.md", "Streaming client sketch (Python, reads X-Sample-Rate)")
def streaming_sketch(ctx: Context, run: Run) -> None:
    def send() -> httpx.Response:
        with ctx.http.stream("POST", "/v1/audio/speech", files=form({"text": "Streaming from python."})) as r:
            r.read()
            return r

    response = retry_busy(send)
    check_pcm(run, response)
    rate = header(response, "x-sample-rate")
    run.check("header.x-sample-rate is an int", True, isinstance(rate, str) and rate.isdigit())


@example(8, "server.md", "POST /v1/audio/speech > Errors: 400 text is required")
def error_text_required(ctx: Context, run: Run) -> None:
    response = post_speech(ctx, {"instruction": "Speak clearly and naturally."})
    run.check("status", 400, response.status_code)
    run.check("body", {"error": "text is required"}, body_json(response))


@example(9, "server.md", "POST /v1/audio/speech > Errors: 404 unknown voice_id (also voices.md > Using one)")
def error_unknown_voice(ctx: Context, run: Run) -> None:
    response = post_speech(ctx, {"text": "Hello there.", "voice_id": f"{run.tmp_name}_missing"})
    run.check("status", 404, response.status_code)
    run.check("body", {"error": "unknown voice_id"}, body_json(response))


@example(10, "server.md", "POST /v1/audio/speech > Errors: 409 busy while another generation runs")
def error_busy(ctx: Context, run: Run) -> None:
    long_text = ("The lighthouse keeper climbed the long spiral stairs every evening, "
                 "counting each step aloud as the last of the daylight faded over the water.")
    for _ in range(BUSY_RETRIES):
        # The stream's headers arrive with its first audio chunk, so from here the GPU is busy.
        with ctx.http.stream("POST", "/v1/audio/speech", files=form({"text": long_text})) as first:
            if first.status_code == 409:
                time.sleep(BUSY_BACKOFF)
                continue
            if first.status_code != 200:
                first.read()
                raise RuntimeError(f"first request got {first.status_code}, not 200 or 409: {first.text[:200]}")
            run.check("first.status", 200, first.status_code)
            second = ctx.http.post("/v1/audio/speech", files=form({"text": "Hello there."}))
            run.check("status", 409, second.status_code)
            run.check("body", {"error": "busy"}, body_json(second))
            return  # leaving the block disconnects, which stops the first generation
    raise RuntimeError("the server stayed busy; could not start the first request")


CORS_PROBE_ORIGIN = "http://cpp-examples.invalid"


def _cors_origin(ctx: Context) -> str:
    """An origin the server allows: any for `--cors`, else the first allowlist entry."""
    if ctx.cors in (None, "*"):
        return CORS_PROBE_ORIGIN
    return ctx.cors.split(",")[0].strip()


EXPOSES_SAMPLE_HEADERS = Pred(
    "a list naming X-Sample-Rate and X-Sample-Format",
    lambda v: isinstance(v, str) and {"x-sample-rate", "x-sample-format"} <= {h.strip().lower() for h in v.split(",")},
)


@example(11, "server.md", "Cross origin requests: CORS headers off by default, on with --cors")
def cors_headers(ctx: Context, run: Run) -> None:
    origin = _cors_origin(ctx)
    response = ctx.http.get("/health", headers={"Origin": origin})
    run.check("status", 200, response.status_code)
    if ctx.cors is None:
        run.check("header.access-control-allow-origin", MISSING, header(response, "access-control-allow-origin"))
        return
    run.check("header.access-control-allow-origin", "*" if ctx.cors == "*" else origin,
              header(response, "access-control-allow-origin"))
    run.check("header.access-control-expose-headers", EXPOSES_SAMPLE_HEADERS,
              header(response, "access-control-expose-headers"))
    if ctx.cors != "*":
        vary = header(response, "vary")
        run.check("header.vary names Origin", True, isinstance(vary, str) and "origin" in vary.lower())


@example(12, "server.md", "Cross origin requests: preflight before DELETE /v1/voices/<id> is 204, cached a day")
def cors_preflight(ctx: Context, run: Run) -> None:
    response = ctx.http.request("OPTIONS", f"/v1/voices/{run.tmp_name}", headers={
        "Origin": _cors_origin(ctx), "Access-Control-Request-Method": "DELETE",
    })
    # The doc describes the 204 only with --cors. Without it the C++ server answers OPTIONS 404
    # (spec.md BC-18), so that is the C++ side to compare; this server's 405 is the BC-18 entry.
    run.check("status", 204 if ctx.cors is not None else 404, response.status_code)
    run.check("header.access-control-max-age", "86400" if ctx.cors is not None else MISSING,
              header(response, "access-control-max-age"))


@example(13, "server.md", "Cross origin requests: an origin not on the --cors list gets no CORS headers")
def cors_not_listed(ctx: Context, run: Run) -> None:
    if ctx.cors in (None, "*"):
        raise Skip("needs a --cors allowlist")
    response = ctx.http.get("/health", headers={"Origin": "http://not-listed.invalid"})
    run.check("status", 200, response.status_code)
    run.check("header.access-control-allow-origin", MISSING, header(response, "access-control-allow-origin"))


# --------------------------------------------------------------------------------------------
# voices.md
# --------------------------------------------------------------------------------------------


@example(14, "voices.md", "Making a saved voice: POST /v1/voices with name=harbour")
def make_saved_voice(ctx: Context, run: Run) -> None:
    try:
        response = register_voice(ctx, run.tmp_name, ctx.ref_text)
        run.check("status", 200, response.status_code)
        run.check("body", voice_shape(run.tmp_name, True, ctx.ref_text), body_json(response))
    finally:
        delete_voice(ctx, run.tmp_name)


@example(15, "voices.md", "Using one: POST /v1/audio/speech with voice_id=harbour")
def use_saved_voice(ctx: Context, run: Run) -> None:
    with temp_voice(ctx, run) as voice_id:
        check_pcm(run, post_speech(ctx, {"text": "It is good to hear your voice again.", "voice_id": voice_id}))


@example(16, "voices.md", "Two kinds of voice: a cached voice posted twice keeps its id")
def cached_voice(ctx: Context, run: Run) -> None:
    # A transcript unique to this run, so the id is new and deleting it removes nobody else's.
    ref_text = f"{ctx.ref_text} ({run.tmp_name} {ctx.run_tag})"
    first = register_voice(ctx, None, ref_text)
    run.check("status", 200, first.status_code)
    first_body = body_json(first)
    run.check("body", voice_shape(STRING, False, ref_text), first_body)
    if first.status_code != 200:
        return
    try:
        second = register_voice(ctx, None, ref_text)
        run.check("second.status", 200, second.status_code)
        second_body = body_json(second)
        second_id = second_body.get("id", MISSING) if isinstance(second_body, dict) else MISSING
        run.check("second.body.id", first_body["id"], second_id)
    finally:
        delete_voice(ctx, first_body["id"])


@example(17, "voices.md", "Listing and removing: GET /v1/voices")
def list_voices(ctx: Context, run: Run) -> None:
    with temp_voice(ctx, run) as voice_id:
        response = ctx.http.get("/v1/voices")
        run.check("status", 200, response.status_code)
        body = body_json(response)
        if not isinstance(body, list):
            run.check("body", "a JSON array", body)
            return
        for index, voice in enumerate(body):
            run.check(f"body.{index}", voice_shape(STRING, BOOL, STRING), voice)
        ours = [v for v in body if isinstance(v, dict) and v.get("id") == voice_id]
        run.check("body[id=tmp]", [voice_shape(voice_id, True, ctx.ref_text)], ours)


@example(18, "voices.md", "Listing and removing: DELETE /v1/voices/<id>")
def delete_saved_voice(ctx: Context, run: Run) -> None:
    with temp_voice(ctx, run) as voice_id:
        response = delete_voice(ctx, voice_id)
        run.check("status", 200, response.status_code)
        run.check("body", {"deleted": voice_id, "file_kept": True}, body_json(response))
        listed = [v.get("id") for v in ctx.http.get("/v1/voices").json()]
        run.check("listed after delete", False, voice_id in listed)
    # temp_voice's own DELETE then gets a 404, which is fine.


# --------------------------------------------------------------------------------------------
# websocket.md
# --------------------------------------------------------------------------------------------


@example(19, "websocket.md", "Connecting: ws_port from /health, then the ready message")
async def ws_connecting(ctx: Context, run: Run) -> None:
    async with ws_open(ctx) as probe:
        check_ready(ctx, run, await probe.until("ready"))


@example(20, "websocket.md", "A session: start, text, instruction, end")
async def ws_session(ctx: Context, run: Run) -> None:
    with temp_voice(ctx, run) as voice_id:
        async with ws_open(ctx) as probe:
            check_ready(ctx, run, await probe.until("ready"))
            await probe.send(type="start", voice_id=voice_id, seed=7)
            await probe.until("started", "error")
            await probe.send(type="text", text="The harbour was quiet that morning. ")
            # The doc sends the instruction while the first piece is being spoken.
            await probe.until("speaking", "error")
            await probe.send(type="instruction", instruction="Speak in an urgent, alarmed whisper.")
            await probe.send(type="end", text="Then the alarm went off.")
            await probe.until("done", "error")
    run.check("events", [
        {"type": "started", "voice_id": voice_id},
        {"type": "speaking", "text": "The harbour was quiet that morning."},
        {"type": "instruction_set"},
        {"type": "speaking", "text": "Then the alarm went off."},
        {"type": "done"},
    ], json_events(probe.items))
    run.check("audio after each speaking", True, audio_after_each_speaking(probe.items))


@example(21, "websocket.md", "Feeding text as it arrives: unfinished text waits, flush speaks it")
async def ws_flush(ctx: Context, run: Run) -> None:
    async with ws_open(ctx) as probe:
        await probe.until("ready")
        await probe.send(type="start")
        await probe.until("started", "error")
        await probe.send(type="text", text="A fragment with no ending")
        await probe.collect_for(1.5)
        run.check("events before flush", [{"type": "started", "voice_id": STRING}], json_events(probe.items))
        await probe.send(type="flush")
        await probe.until("speaking", "error")
        await probe.send(type="end")
        await probe.until("done", "error")
    run.check("events", [
        {"type": "started", "voice_id": STRING},
        {"type": "speaking", "text": "A fragment with no ending"},
        {"type": "done"},
    ], json_events(probe.items))
    run.check("audio after each speaking", True, audio_after_each_speaking(probe.items))


@example(22, "websocket.md", "Interrupting: cancel stops the piece, the session carries on")
async def ws_cancel(ctx: Context, run: Run) -> None:
    async with ws_open(ctx) as probe:
        await probe.until("ready")
        await probe.send(type="start")
        await probe.until("started", "error")
        await probe.send(type="text", text="This sentence is long enough that the cancel lands while it is still being spoken aloud. ")
        await probe.until("speaking", "error")
        await probe.until(AUDIO, "error")
        await probe.send(type="cancel")
        await probe.until("cancelled", "error")
        await probe.send(type="end", text="Then it carried on.")
        await probe.until("done", "error")
    run.check("events", [
        {"type": "started", "voice_id": STRING},
        {"type": "speaking", "text": "This sentence is long enough that the cancel lands while it is still being spoken aloud."},
        {"type": "cancelled"},
        {"type": "speaking", "text": "Then it carried on."},
        {"type": "done"},
    ], json_events(probe.items))
    run.check("audio after each speaking", True, audio_after_each_speaking(probe.items))


@example(23, "websocket.md", "Concurrency: a second session waits its turn and reports queued")
async def ws_queued(ctx: Context, run: Run) -> None:
    async with ws_open(ctx) as first, ws_open(ctx) as second:
        for probe in (first, second):
            await probe.until("ready")
            await probe.send(type="start")
            await probe.until("started", "error")
        await first.send(type="end", text="The first session speaks this sentence slowly, from the beginning of the line to the very end of it.")
        await first.until(AUDIO, "error")
        await second.send(type="end", text="The second one waits.")
        await second.until("done", "error")
        await first.until("done", "error")
    run.check("second.events", [
        {"type": "started", "voice_id": STRING},
        {"type": "queued"},
        {"type": "speaking", "text": "The second one waits."},
        {"type": "done"},
    ], json_events(second.items, drop=("ready",)))


@example(24, "websocket.md", "Messages you receive: error with message (start with an unknown voice_id)")
async def ws_error(ctx: Context, run: Run) -> None:
    async with ws_open(ctx) as probe:
        await probe.until("ready")
        await probe.send(type="start", voice_id=f"{run.tmp_name}_missing")
        await probe.until("error", "started")
    run.check("events", [{"type": "error", "message": STRING}], json_events(probe.items))


@example(25, "websocket.md", "Client sketch: start, text lines, end with empty text, read until done")
async def ws_client_sketch(ctx: Context, run: Run) -> None:
    with temp_voice(ctx, run) as voice_id:
        async with ws_open(ctx) as probe:
            await probe.until("ready")
            await probe.send(type="start", voice_id=voice_id, seed=7)
            for line in ["It is good to hear your voice again."]:
                await probe.send(type="text", text=line)
            await probe.send(type="end", text="")
            last = await probe.until("done", "error")
    run.check("last event", {"type": "done"}, last)
    run.check("audio.bytes", PCM_BODY, probe.audio_bytes)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def run_example(ctx: Context, ex: Example) -> ExampleResult:
    result = ExampleResult(ex.example_id, ex.title)
    run = Run(ex.example_id, ex.number)
    started = time.monotonic()
    try:
        outcome_or_coroutine = ex.fn(ctx, run)
        if inspect.iscoroutine(outcome_or_coroutine):
            asyncio.run(outcome_or_coroutine)
    except Skip as skip:
        result.skipped = str(skip)
    except Exception as error:  # noqa: BLE001 -- an example that can't finish is a failure, never a crash
        result.error = f"{type(error).__name__}: {error}"
    result.diffs = run.diffs
    result.notes = [*run.notes, f"{time.monotonic() - started:.1f} s"]
    return result


def discover(ctx: Context, base_url: str) -> None:
    """Read the model rate and the WebSocket port from /health, as websocket.md says to."""
    body = ctx.http.get("/health").json()
    ctx.sample_rate = body["sample_rate"]
    if body.get("ws_port"):
        host = httpx.URL(base_url).host
        if ":" in host:  # an IPv6 literal needs brackets in a URL
            host = f"[{host}]"
        ctx.ws_url = f"ws://{host}:{body['ws_port']}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="the server's HTTP base URL")
    parser.add_argument("--cors", nargs="?", const="*", default=None, metavar="ORIGINS",
                        help="the --cors option the server was launched with (omit when it had none)")
    parser.add_argument("--ref-wav", type=Path, default=DEFAULT_REF_WAV,
                        help="reference clip used as upload content; its transcript is the .txt beside it")
    parser.add_argument("--only", default="", help="comma-separated example numbers, e.g. 3,20")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    transcript = args.ref_wav.with_suffix(".txt")
    if not args.ref_wav.is_file() or not transcript.is_file():
        print(f"reference clip or transcript missing: {args.ref_wav}, {transcript}", file=sys.stderr)
        return 2
    only = {int(n) for n in args.only.split(",") if n.strip()}
    examples = sorted((e for e in EXAMPLES if not only or e.number in only), key=lambda e: e.number)

    # trust_env=False: never route a local server through an environment proxy.
    with httpx.Client(base_url=args.url, timeout=HTTP_TIMEOUT, trust_env=False) as http:
        ctx = Context(http=http, ref_wav=args.ref_wav.read_bytes(), ref_text=transcript.read_text().strip(),
                      cors=args.cors, run_tag=uuid.uuid4().hex[:8])
        try:
            discover(ctx, args.url)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            print(f"GET {args.url}/health failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(f"C++ doc examples against {args.url} (WebSocket {ctx.ws_url or 'disabled'}), "
              f"cors={args.cors or 'off'}, docs in {C_DOCS}")
        leftovers = sweep(ctx)
        if leftovers:
            print(f"removed leftover throwaway voices: {', '.join(leftovers)}")
        results: list[ExampleResult] = []
        try:
            for ex in examples:
                result = run_example(ctx, ex)
                results.append(result)
                print(f"  {ex.example_id} {outcome(result)}", flush=True)
        finally:
            try:
                sweep(ctx)
            except Exception as error:  # noqa: BLE001 -- a bad body from the sweep must not lose the report
                print(f"WARNING: the closing sweep failed ({error}); delete {TMP_PREFIX}* voices by hand",
                      file=sys.stderr)
    print()
    print("\n".join(render_report(results)))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
