"""Long text: splitting, anchoring and per-piece room (specs/003-cpp-compatible-api/tasks.md
T050; spec US3, FR-013, FR-014, FR-036a, BC-47; data-model.md "Reference" and "Piece").

The anchoring cases are ported from `A:tests/test_api_speech.py` (~462-494). Every test runs
the real route (`api.create_app`) over `tests/fakes.py`'s `FakeRuntime`; the one that needs to
see how a stream ends on the wire runs a real uvicorn, as `tests/test_speech_abort.py` does.

Which reference each piece was prepared with is read off a spy on `routes_speech.prepare_piece`:
`FakeTokenizer.decode` renders every text segment as `x`s, so `ref_text` can't be recovered
from the runtime's `inputs` themselves (the anchor's codes can, from `input_values`).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import torch

from breeze_infer import limits, routes_speech
from breeze_infer.api import Components, create_app
from breeze_infer.gpu import GPU_CLOSE_TIMEOUT_SECONDS, GpuGate, GpuUnavailable
from breeze_infer.limits import ANCHOR_CHARS
from breeze_infer.routes_health import Readiness
from breeze_infer.synthesis import (
    CodesRef,
    NoRef,
    PieceRoom,
    anchor_codes,
    prepare_piece,
)
from breeze_infer.text_split import split_text
from tests.fakes import (
    CODEC_CODEBOOKS,
    FakeStreamingConfig,
    FakeTokenizer,
    RecordingEvents,
    codec_frame_count,
    model_with_codec_facts,
)
from tests.test_routes_speech import (
    SPEECH_PATH,
    _build_components,
    _client_for,
    _fake_runtime,
    _wav_bytes,
)
from tests.test_routes_speech import _gate_is_free as _components_gate_is_free
from tests.test_speech_abort import LiveServer, eventually, wait_until
from tests.test_validation_order import _serve, _text_request

PAD = 2050  # fake_model()'s codebook_pad_token_id, as the real checkpoint's
INSTRUCTION = "Speak calmly."

# Six sentences of about 60 characters: with split_chars=100, every piece is one of them.
SENTENCES = " ".join(
    f"Sentence number {n} is here to fill out the long passage well." for n in range(6)
)


def _no_reference_pieces(text: str, split_chars: int) -> list[str]:
    """How the route splits `text` with no reference: the opening budget, capped at the
    piece budget so piece 0 is never the largest (review 26b #3)."""
    return split_text(text, budget=split_chars, first_budget=min(ANCHOR_CHARS, split_chars))


def _frame(value: int) -> torch.Tensor:
    """One generated frame as the runtime's `token_observer` sees it: every codebook's code."""
    return torch.full((CODEC_CODEBOOKS,), value, dtype=torch.long)


def _events(events: RecordingEvents, name: str) -> list[dict[str, object]]:
    return [fields for event, fields in events.calls if event == name]


@pytest.fixture()
def prepared(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every `prepare_piece` call the route makes: the reference and text of each piece."""
    calls: list[dict[str, Any]] = []

    def spy(tokenizer, model, reference, text, instruction, cfg_scale):
        calls.append({"reference": reference, "text": text})
        return prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)

    monkeypatch.setattr(routes_speech, "prepare_piece", spy)
    return calls


class Env:
    """A ready app over a `FakeRuntime` whose GPU thread is shut down afterwards."""

    def __init__(self, runtime: Any, *, split_chars: int | None = None) -> None:
        self.readiness = Readiness()
        self.events = RecordingEvents()
        self.components: Components = _build_components(
            self.readiness, split_chars=split_chars, events=self.events
        )
        self.runtime = runtime
        self.readiness.mark_ready(runtime)
        self.client = _client_for(self.components)

    def speak(self, **fields: str) -> httpx.Response:
        data = {"instruction": INSTRUCTION, **fields}
        return self.client.post(SPEECH_PATH, data=data)


@pytest.fixture()
def envs() -> Iterator[list[Env]]:
    started: list[Env] = []
    yield started
    for env in started:
        env.components.gpu.shutdown()


def _env(envs: list[Env], runtime: Any, **kwargs: Any) -> Env:
    env = Env(runtime, **kwargs)
    envs.append(env)
    return env


def _seq_len(reference: Any, text: str) -> int:
    """The prompt length of one piece as the route builds it (cfg_scale 1: one row)."""
    inputs = prepare_piece(
        FakeTokenizer(), model_with_codec_facts(), reference, text, INSTRUCTION, 1.0
    )
    return int(inputs["attention_mask"].shape[1])


# --- anchor_codes ---------------------------------------------------------------------------


def test_anchor_codes_drops_all_pad_frames_and_stacks_the_rest() -> None:
    partly_pad = _frame(9)
    partly_pad[3] = PAD

    codes = anchor_codes([_frame(5), _frame(PAD), partly_pad, _frame(7)], PAD)

    assert codes is not None
    assert codes.shape == (3, CODEC_CODEBOOKS)
    assert codes[:, 0].tolist() == [5, 9, 7]  # only an all-pad frame is dropped


def test_anchor_codes_is_none_when_nothing_but_pad_was_generated() -> None:
    assert anchor_codes([_frame(PAD), _frame(PAD)], PAD) is None
    assert anchor_codes([], PAD) is None


# --- anchoring ------------------------------------------------------------------------------


def test_first_piece_anchors_every_later_piece(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(PAD), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    pieces = _no_reference_pieces(SENTENCES, 100)
    assert len(pieces) >= 3

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    assert [p["text"] for p in prepared] == pieces
    assert isinstance(prepared[0]["reference"], NoRef)
    for later in prepared[1:]:
        reference = later["reference"]
        assert isinstance(reference, CodesRef)
        assert reference.ref_text == pieces[0]
        assert reference.codes[:, 0].tolist() == [5, 7]  # the all-pad frame is dropped
    # The anchor reaches the runtime: every later piece's inputs carry piece 0's codes.
    for call in runtime.calls[1:]:
        assert call["inputs"]["input_values"][0, :, 0].tolist() == [5, 7]
    # Only piece 0 keeps its frames; once anchored, later pieces don't.
    assert runtime.calls[0]["observed"] is True
    assert all(call["observed"] is False for call in runtime.calls[1:])


def test_later_pieces_are_prepared_one_at_a_time_as_the_stream_reaches_them(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor's room check runs before piece 1, but no later piece's model inputs are
    built (or kept) ahead of time: each is prepared once, when the loop gets to it."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    started_before: list[tuple[str, int]] = []

    def spy(tokenizer, model, reference, text, instruction, cfg_scale):
        started_before.append((text, len(runtime.calls)))
        return prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)

    monkeypatch.setattr(routes_speech, "prepare_piece", spy)
    env = _env(envs, runtime, split_chars=100)
    pieces = _no_reference_pieces(SENTENCES, 100)
    assert len(pieces) >= 3

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    assert _events(env.events, "speech.anchor_skipped") == []
    # Piece i is prepared once, after the i pieces before it have started generating.
    assert started_before == [(text, index) for index, text in enumerate(pieces)]


def test_later_pieces_are_sized_after_the_lease_on_the_cpu_executor(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor decision needs every later piece's length. Those are measured once, on the
    sizing worker's own thread (review 34 finding 5), and -- since review #2 on 2d9070a moved
    it -- only after the lease is taken, not before the busy check: a request that is about to
    get a `409` must never size its whole text."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    sized: list[tuple[str, bool, list[str]]] = []
    real_sizing = routes_speech.anchor_sizing

    def spy(sizing_runtime, tokenizer, anchor_text, later_texts, instruction, cfg_scale):
        # Read-only, from the executor thread: the gate itself belongs to the event loop.
        gate_free = env.components.gate._owner is None
        sized.append((threading.current_thread().name, gate_free, list(later_texts)))
        return real_sizing(
            sizing_runtime, tokenizer, anchor_text, later_texts, instruction, cfg_scale
        )

    monkeypatch.setattr(routes_speech, "anchor_sizing", spy)
    pieces = _no_reference_pieces(SENTENCES, 100)

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    [(thread, gate_free, later)] = sized
    assert thread.startswith("breeze-anchor-sizing")
    assert not gate_free  # after the lease: this request's own lease still holds it
    assert later == pieces[1:]


def test_a_busy_request_never_sizes_the_later_pieces(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that gets `409 busy` never reaches `_start_anchor_sizing` at all (review #2
    on 2d9070a): the lease it would have been queued under is never acquired."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    sized: list[Any] = []
    real_sizing = routes_speech.anchor_sizing

    def spy(*args: Any, **kwargs: Any) -> Any:
        sized.append(args)
        return real_sizing(*args, **kwargs)

    monkeypatch.setattr(routes_speech, "anchor_sizing", spy)
    lease = env.components.gate.try_acquire()
    assert lease is not None

    response = env.speak(text=SENTENCES)

    assert response.status_code == 409
    assert sized == []


def test_the_anchor_decision_waits_for_a_slow_sizing_result(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queuing the sizing right after the lease (review #2 on 2d9070a) lets it run on the CPU
    while piece 0 itself prepares and generates on the GPU thread -- but if it is still
    running once piece 0 finishes, `_anchor_for_later_pieces` must block for the real result
    rather than press on without it. Proven with an artificially slow `anchor_sizing`: the
    request still completes, correctly anchored, and takes at least as long as the delay."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    real_sizing = routes_speech.anchor_sizing
    delay_seconds = 0.2

    def slow_sizing(sizing_runtime, tokenizer, anchor_text, later_texts, instruction, cfg_scale):
        time.sleep(delay_seconds)
        return real_sizing(
            sizing_runtime, tokenizer, anchor_text, later_texts, instruction, cfg_scale
        )

    monkeypatch.setattr(routes_speech, "anchor_sizing", slow_sizing)

    started = time.monotonic()
    response = env.speak(text=SENTENCES)
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed >= delay_seconds  # the anchor decision actually waited for the real result
    assert _events(env.events, "speech.anchor_skipped") == []


class _CallRecordingTokenizer(FakeTokenizer):
    """Records how many pieces had started generating (`runtime.calls`) at each call."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self.started_at: list[int] = []

    def __call__(self, text: str, **kwargs: Any) -> Any:
        self.started_at.append(len(self._runtime.calls))
        return super().__call__(text, **kwargs)


def test_between_piece_0_and_piece_1_only_piece_1_is_tokenized_on_the_gpu_thread(
    envs: list[Env],
) -> None:
    """With the lengths measured before the gate, the anchor check after piece 0 is arithmetic:
    the GPU thread's tokenizer only builds piece 1's own inputs before piece 1 starts."""
    frames = [_frame(5), _frame(6), _frame(7)]
    runtime = _fake_runtime(chunks=3, frames=frames)
    gpu_tokenizer = _CallRecordingTokenizer(runtime)
    runtime.tokenizer = gpu_tokenizer
    env = _env(envs, runtime, split_chars=100)
    pieces = _no_reference_pieces(SENTENCES, 100)
    assert len(pieces) > 3
    counting = _CallRecordingTokenizer(runtime)
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    prepare_piece(counting, model_with_codec_facts(), anchor, pieces[1], INSTRUCTION, 1.0)

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    assert _events(env.events, "speech.anchor_skipped") == []
    between_0_and_1 = gpu_tokenizer.started_at.count(1)
    assert between_0_and_1 == len(counting.started_at)


def test_no_anchor_when_piece_0_produced_no_frames(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    # Every frame is all-pad, so piece 0 leaves nothing to anchor on. Later pieces stay voice
    # design, and none of them becomes the anchor instead (data-model.md: only piece 0 can).
    runtime = _fake_runtime(chunks=1, frames=[_frame(PAD)])
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    assert len(prepared) >= 3
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    assert all(call["observed"] is False for call in runtime.calls[1:])


def test_a_single_piece_keeps_no_frames(envs: list[Env]) -> None:
    runtime = _fake_runtime(chunks=2, frames=[_frame(5), _frame(6)])
    env = _env(envs, runtime)

    response = env.speak(text="Just one short piece.")

    assert response.status_code == 200
    assert [call["observed"] for call in runtime.calls] == [False]


def test_with_a_reference_every_piece_uses_that_reference(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    runtime = _fake_runtime(chunks=2, frames=[_frame(5), _frame(6)])
    env = _env(envs, runtime, split_chars=100)

    response = env.client.post(
        SPEECH_PATH,
        data={"text": SENTENCES, "ref_text": "a reference transcript"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    assert [p["text"] for p in prepared] == split_text(SENTENCES, budget=100)
    first = prepared[0]["reference"]
    assert isinstance(first, CodesRef)
    assert first.ref_text == "a reference transcript"
    assert all(p["reference"] is first for p in prepared)
    assert all(call["observed"] is False for call in runtime.calls)
    assert runtime.audio_tokenizer.encode_calls == 1


def test_the_opening_budget_applies_only_without_a_reference(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    text = " ".join(f"Line {n} of the voice test text." for n in range(40))
    env = _env(envs, _fake_runtime(chunks=1, frames=[_frame(5)]), split_chars=600)

    assert env.speak(text=text).status_code == 200
    no_reference = [p["text"] for p in prepared]
    prepared.clear()
    response = env.client.post(
        SPEECH_PATH,
        data={"text": text, "ref_text": "a reference transcript"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    with_reference = [p["text"] for p in prepared]

    assert response.status_code == 200
    assert no_reference == _no_reference_pieces(text, 600)
    assert with_reference == split_text(text, budget=600)
    assert no_reference[0] != with_reference[0]  # only the first piece is packed differently


def test_split_chars_zero_gives_one_piece(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    runtime = _fake_runtime(chunks=1, frames=[_frame(5)])
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES, split_chars="0")

    assert response.status_code == 200
    assert [p["text"] for p in prepared] == [SENTENCES]
    assert len(runtime.calls) == 1


# --- room (BC-47, FR-036a) --------------------------------------------------------------------


def test_bc_47_first_piece_without_room_gets_400_text_too_long(envs: list[Env]) -> None:
    """BC-47: a first piece that cannot fit the model's context gets `400 text_too_long`
    -- the C++ server had no fixed context limit, so a piece of any length could generate
    (or exhaust the cache it sized per piece).
    """
    runtime = _fake_runtime(config=FakeStreamingConfig(max_seq_len=1))
    env = _env(envs, runtime)

    response = env.speak(text="Not enough room for this to generate at all.")

    assert response.status_code == 400
    assert response.json() == {"error": "text is too long", "code": "text_too_long"}
    assert runtime.calls == []


def test_first_piece_without_room_is_400_even_while_the_gpu_is_busy(envs: list[Env]) -> None:
    """FR-007: every 400 comes before 409 busy, so piece 0's room is checked before the gate
    is taken (on the tokenized text plus the reference's predicted frames)."""
    runtime = _fake_runtime(config=FakeStreamingConfig(max_seq_len=1))
    env = _env(envs, runtime)
    held = env.components.gate.try_acquire()
    assert held is not None
    try:
        no_reference = env.speak(text="Not enough room for this to generate at all.")
        inline = env.client.post(
            SPEECH_PATH,
            data={"text": "Not enough room.", "ref_text": "a reference transcript"},
            files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    finally:
        held.release()

    assert no_reference.status_code == 400
    assert no_reference.json()["code"] == "text_too_long"
    assert inline.status_code == 400
    assert inline.json()["code"] == "text_too_long"
    assert runtime.audio_tokenizer.encode_calls == 0  # rejected before the codec ran


def _gate_is_free(gate: GpuGate, server: LiveServer) -> bool:
    def probe() -> bool:
        lease = gate.try_acquire()
        if lease is None:
            return False
        lease.release()
        return True

    return server.call(probe)


def test_bc_47_later_piece_without_room_aborts_the_stream(envs: list[Env]) -> None:
    """Piece 0 fits; piece 1, a much longer sentence with the same reference, doesn't. The
    200 is already out by then, so the stream ends without the chunked terminator (FR-013).
    A reference, not an anchor: an anchor that leaves no room is skipped instead (#4)."""
    wav = _wav_bytes()
    reference = CodesRef(
        codes=torch.zeros((codec_frame_count(8000, 16000), CODEC_CODEBOOKS), dtype=torch.int16),
        ref_text="a reference transcript",
    )
    long_sentence = " ".join(["This long sentence keeps going"] * 9) + "."
    text = f"Short one. {long_sentence}"
    pieces = split_text(text, budget=285)
    assert pieces == ["Short one.", long_sentence]
    # Piece 0 has room for its frames; piece 1's prompt fills the context exactly.
    max_seq_len = _seq_len(reference, pieces[1]) + 1
    assert max_seq_len - _seq_len(reference, pieces[0]) - 1 >= 2
    runtime = _fake_runtime(chunks=2, config=FakeStreamingConfig(max_seq_len=max_seq_len))
    env = _env(envs, runtime)
    server = LiveServer(create_app(env.components))
    try:
        received = bytearray()
        url = f"http://127.0.0.1:{server.port}{SPEECH_PATH}"
        data = {
            "text": text,
            "instruction": INSTRUCTION,
            "ref_text": reference.ref_text,
            "split_chars": "285",
        }
        files = {"ref_audio": ("ref.wav", wav, "audio/wav")}
        with (
            httpx.Client(timeout=10) as client,
            client.stream("POST", url, data=data, files=files) as response,
        ):
            assert response.status_code == 200
            with pytest.raises(httpx.RemoteProtocolError):
                for data_chunk in response.iter_raw():
                    received += data_chunk

        assert len(received) > 0  # piece 0's audio went out before the failure
        assert len(runtime.calls) == 1  # piece 1 never started
        wait_until(lambda: _events(env.events, "speech.failed") != [])
        server.wait_for_handlers()
        assert _events(env.events, "speech.completed") == []
        assert _gate_is_free(env.components.gate, server)
    finally:
        server.stop()


# Piece 0 is short; every later piece is one long sentence of the same length.
SHORT_THEN_LONG = "Short one. " + " ".join(
    f"Long sentence number {n} keeps going for quite a while, on and on, "
    "to fill out its whole piece."
    for n in range(4)
)


def test_an_anchor_clamped_more_than_without_it_is_skipped(envs: list[Env]) -> None:
    """The broader `no_room` rule (decided with the user, 2026-09-25): piece 1 already has
    less than its cap *without* the anchor -- its own long sentence fills most of the
    context -- and the anchor would leave it with even less room, not the same. The old
    rule only compared an anchored room to the cap, so a piece already clamped without the
    anchor never even entered the comparison; it would have kept the anchor here and let it
    shrink an already-clamped piece further. Now the anchor is skipped
    (`speech.anchor_skipped`, `no_room`), and every later piece is generated -- and clamped
    -- at the room it has without the anchor, exactly as `speech.piece_clamped` records it.

    `chunks` (the fake's own natural completion length, decoupled here from the anchor's own
    4 frames -- `FakeRuntime` only ever observes as many real frames as `frames` holds,
    whatever `chunks` is) is picked comfortably above every room below, so it is the room,
    not the fake just running out of frames on its own, that actually clamps each piece.
    """
    frames = [_frame(n) for n in range(1, 5)]
    pieces = _no_reference_pieces(SHORT_THEN_LONG, 100)
    assert pieces[0] == "Short one." and len(pieces) == 5
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    room_with_anchor = 15
    max_seq_len = _seq_len(anchor, pieces[1]) + 1 + room_with_anchor
    room_without_anchor = max_seq_len - _seq_len(NoRef(), pieces[1]) - 1
    assert room_without_anchor > room_with_anchor  # shorter with the anchor, not the same
    chunks = 300
    runtime = _fake_runtime(
        chunks=chunks, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SHORT_THEN_LONG)

    assert response.status_code == 200
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "no_room"
    clamped = {e["piece_index"]: e for e in _events(env.events, "speech.piece_clamped")}
    for index in range(1, len(pieces)):
        assert clamped[index]["room"] == room_without_anchor
    assert clamped[1]["request_id"] == response.headers["x-request-id"]
    assert runtime.calls[1]["max_new_tokens"] == room_without_anchor
    assert runtime.calls[1]["inputs"]["input_values"] is None  # not anchored
    done = {e["piece_index"]: e["frames"] for e in _events(env.events, "speech.piece_done")}
    for index in range(1, len(pieces)):
        assert done[index] == room_without_anchor
    assert len(_events(env.events, "speech.completed")) == 1


def test_an_anchor_that_would_leave_an_already_clamped_piece_no_room_is_skipped(
    envs: list[Env],
) -> None:
    """The gap the old floor missed (decided with the user, 2026-09-25): piece 1 is already
    clamped to a small room *without* the anchor; the anchor's own frames would use up the
    rest of the context, leaving it zero or less. The old rule never compared this piece's
    anchored room to anything -- it only checked pieces that had their full cap without the
    anchor -- so it would have kept the anchor, driven piece 1's room to zero or below, and
    (BC-47) aborted the stream after the `200` instead of completing it.
    """
    frames = [_frame(n) for n in range(1, 5)]
    pieces = _no_reference_pieces(SHORT_THEN_LONG, 100)
    assert pieces[0] == "Short one." and len(pieces) == 5
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    room_without_anchor = 5
    max_seq_len = _seq_len(NoRef(), pieces[1]) + 1 + room_without_anchor
    room_with_anchor = max_seq_len - _seq_len(anchor, pieces[1]) - 1
    assert room_with_anchor <= 0  # the anchor's own frames use up the rest of the context
    chunks = 300  # comfortably above room_without_anchor, so room -- not the fake's own
    # natural completion -- is what clamps piece 1 (see the other test's own docstring).
    runtime = _fake_runtime(
        chunks=chunks, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SHORT_THEN_LONG)

    assert response.status_code == 200
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "no_room"
    clamped = {e["piece_index"]: e["room"] for e in _events(env.events, "speech.piece_clamped")}
    for index in range(1, len(pieces)):
        assert clamped[index] == room_without_anchor
    assert runtime.calls[1]["max_new_tokens"] == room_without_anchor
    assert runtime.calls[1]["inputs"]["input_values"] is None  # not anchored
    assert len(_events(env.events, "speech.completed")) == 1
    assert _events(env.events, "speech.failed") == []


# --- when piece 0 is not used as the anchor (review 26b #2, #4) --------------------------------


def test_a_truncated_piece_0_is_not_an_anchor(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    """Piece 0 stopped at its cap, not at EOS: its audio may end mid-word, so it doesn't
    anchor, and the later pieces stay voice design."""
    runtime = _fake_runtime(chunks=10, frames=[_frame(n) for n in range(1, 11)])
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES, max_new_tokens="3")

    assert response.status_code == 200
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "piece_truncated"
    assert skipped["piece_index"] == 0
    assert skipped["request_id"] == response.headers["x-request-id"]


def test_an_anchor_that_would_clamp_a_piece_that_otherwise_fits_is_skipped(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    """With piece 0 as its reference, piece 1 would get less than its cap (30 of 50 frames),
    while without the anchor it gets all 50. The anchor is skipped whole (never trimmed: its
    codes must match its text), and the later pieces are voice design at their full cap. The
    30 frames it would have left are well above the old 12-frame threshold."""
    frames = [_frame(n) for n in range(1, 5)]
    pieces = _no_reference_pieces(SENTENCES, 100)
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    cap = 50
    max_seq_len = _seq_len(anchor, pieces[1]) + 1 + 30
    assert max_seq_len - _seq_len(NoRef(), pieces[1]) - 1 >= cap
    runtime = _fake_runtime(
        chunks=4, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES, max_new_tokens=str(cap))

    assert response.status_code == 200
    assert [p["text"] for p in prepared] == pieces
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "no_room"
    assert _events(env.events, "speech.piece_clamped") == []
    assert [call["max_new_tokens"] for call in runtime.calls[1:]] == [cap] * (len(pieces) - 1)
    assert len(_events(env.events, "speech.completed")) == 1


def test_an_anchor_that_leaves_every_piece_its_full_cap_is_kept(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    """The boundary of the rule: with the anchor, piece 1 still gets exactly its cap."""
    frames = [_frame(n) for n in range(1, 5)]
    pieces = _no_reference_pieces(SENTENCES, 100)
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    cap = 50
    max_seq_len = _seq_len(anchor, pieces[1]) + 1 + cap
    runtime = _fake_runtime(
        chunks=4, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES, max_new_tokens=str(cap))

    assert response.status_code == 200
    assert _events(env.events, "speech.anchor_skipped") == []
    assert all(isinstance(p["reference"], CodesRef) for p in prepared[1:])
    assert _events(env.events, "speech.piece_clamped") == []


def test_split_chars_below_the_opening_budget_does_not_make_piece_0_the_largest(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    text = " ".join(f"Short sentence {n} of the long text here." for n in range(12))
    env = _env(envs, _fake_runtime(chunks=1, frames=[_frame(5)]), split_chars=100)

    response = env.speak(text=text)

    assert response.status_code == 200
    lengths = [len(p["text"]) for p in prepared]
    assert len(lengths) > 2
    assert lengths[0] <= 100
    assert lengths[0] <= max(lengths[1:])


# --- the anchor sizing future on every exit path (review 33 on 10f0c29) ----------------------
#
# `_start_anchor_sizing` queues the later pieces' sizing on the CPU tokenizer's sizing worker
# right after the lease. A request that ends without consuming it must cancel it: abandoned
# sizing of a 10k-character text would otherwise hold that worker, and the next lease holder's
# own sizing would queue behind it. And the GPU thread's `.result()` must never abort a stream
# that is already out, nor stall it for long.


class _SizingHeldInQueue:
    """Keeps the request's anchor sizing *queued*, not running: right before
    `_start_anchor_sizing` queues it, the CPU tokenizer's sizing worker is given a call that waits
    until `release()`. Records every real `anchor_sizing` call, so a test can tell whether the
    sizing ever ran once the worker was free again."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.futures: list[Any] = []
        self.sized: list[Any] = []
        self._released = threading.Event()
        real_start = routes_speech._start_anchor_sizing
        real_sizing = routes_speech.anchor_sizing

        def start(runtime: Any, cpu_tokenizer: Any, request: Any, pieces: list[str]) -> Any:
            if not self._released.is_set():
                cpu_tokenizer.submit(lambda _tokenizer: self._released.wait(5))
            job = real_start(runtime, cpu_tokenizer, request, pieces)
            if job is not None:  # `None`: a single piece, nothing to size
                self.futures.append(job.future)
            return job

        def spy(*args: Any) -> Any:
            self.sized.append(args)
            return real_sizing(*args)

        monkeypatch.setattr(routes_speech, "_start_anchor_sizing", start)
        monkeypatch.setattr(routes_speech, "anchor_sizing", spy)

    def release(self) -> None:
        self._released.set()


def _no_room_on_the_gpu(runtime: Any, reference: Any, text: str, request: Any) -> Any:
    """Piece 0 fits by the CPU's prediction but not once prepared on the GPU thread: the
    `400 text_too_long` the route raises after the lease (a codec that encodes more frames
    than predicted)."""
    inputs = prepare_piece(
        runtime.tokenizer, runtime.model, reference, text, request.instruction, request.cfg_scale
    )
    return inputs, PieceRoom(cap=runtime.frame_cap(request.max_new_tokens), room=0)


def _prepare_fails(runtime: Any, reference: Any, text: str, request: Any) -> Any:
    raise RuntimeError("piece 0 preparation failed")


@pytest.mark.parametrize(
    ("prepare_first_piece", "status"), [(_no_room_on_the_gpu, 400), (_prepare_fails, 500)]
)
def test_a_request_that_fails_before_the_200_cancels_its_queued_sizing(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch, prepare_first_piece: Any, status: int
) -> None:
    """Finding 1: the `400 text_too_long` after the lease, or a failing preparation, ends the
    request with its sizing still queued. It is cancelled, so it never runs: nothing is left
    on the sizing worker for the next lease holder's own sizing to wait behind."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    held = _SizingHeldInQueue(monkeypatch)
    real_prepare_first_piece = routes_speech._prepare_first_piece
    monkeypatch.setattr(routes_speech, "_prepare_first_piece", prepare_first_piece)
    try:
        response = env.speak(text=SENTENCES)
        assert response.status_code == status
        [future] = held.futures
        assert future.cancelled()
    finally:
        held.release()

    monkeypatch.setattr(routes_speech, "_prepare_first_piece", real_prepare_first_piece)
    next_response = env.speak(text="Short one.")  # one piece: nothing of its own to size

    assert next_response.status_code == 200
    assert held.sized == []  # the failed request's sizing never ran on the worker


def test_a_disconnect_after_the_200_cancels_the_queued_sizing(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: a client that leaves while piece 0 is still generating ends the stream with
    the sizing still queued. Closing the generator (`gen.close()` on the GPU thread) cancels
    it, so it never runs on the worker."""
    hold = threading.Event()
    runtime = _fake_runtime(chunks=10, gate=hold, gate_at=2)
    env = _env(envs, runtime, split_chars=100)
    held = _SizingHeldInQueue(monkeypatch)
    server = LiveServer(create_app(env.components))
    try:
        url = f"http://127.0.0.1:{server.port}{SPEECH_PATH}"
        data = {"text": SENTENCES, "instruction": INSTRUCTION}
        with (
            httpx.Client(timeout=10) as client,
            client.stream("POST", url, data=data) as response,
        ):
            assert response.status_code == 200
            next(response.iter_raw())
        # Leaving both blocks closed the connection while piece 0 was still generating.
        hold.set()
        wait_until(lambda: _events(env.events, "speech.aborted") != [])
        server.wait_for_handlers()

        [future] = held.futures
        assert future.cancelled()
    finally:
        hold.set()
        held.release()
        server.stop()
    assert held.sized == []


@pytest.mark.parametrize(
    ("frames", "max_new_tokens"),
    [
        # Piece 0 uses its whole 3-frame limit: `piece_truncated`.
        ([_frame(n) for n in range(1, 11)], "3"),
        # Piece 0 finishes, but every frame is pad: nothing to anchor on.
        ([_frame(PAD)], None),
    ],
    ids=["piece_truncated", "all_pad"],
)
def test_an_anchor_decided_without_the_sizing_cancels_it_before_piece_1(
    envs: list[Env],
    monkeypatch: pytest.MonkeyPatch,
    frames: list[torch.Tensor],
    max_new_tokens: str | None,
) -> None:
    """Review 34 finding 1: both early answers leave the sizing unread. It is cancelled right
    there, before piece 1 is prepared, not only once every later piece has streamed
    (`_iter_pieces`' own `finally`): until then it would hold the sizing worker for nothing."""
    runtime = _fake_runtime(chunks=1, frames=frames)
    env = _env(envs, runtime, split_chars=100)
    held = _SizingHeldInQueue(monkeypatch)
    cancelled_at_piece: list[bool] = []

    def spy(tokenizer, model, reference, text, instruction, cfg_scale):
        if held.futures:
            cancelled_at_piece.append(held.futures[0].cancelled())
        return prepare_piece(tokenizer, model, reference, text, instruction, cfg_scale)

    monkeypatch.setattr(routes_speech, "prepare_piece", spy)
    fields = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
    try:
        response = env.speak(text=SENTENCES, **fields)
    finally:
        held.release()

    assert response.status_code == 200
    # Still queued while piece 0 is prepared; cancelled by the time any later piece is.
    assert len(cancelled_at_piece) == len(_no_reference_pieces(SENTENCES, 100))
    assert cancelled_at_piece[0] is False
    assert all(cancelled_at_piece[1:])
    assert held.sized == []


def test_a_request_cancelled_while_the_gpu_thread_waits_on_its_sizing_reports_no_skip(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 34 finding 3: the route's own cancel (`_serve_speech`'s cleanup, here for a
    request cancelled while priming) can reach the sizing while the GPU thread's in-flight step
    is still waiting on it. That is the request ending, not a shutdown: the stream is over
    anyway, so no `speech.anchor_skipped` is emitted at all -- in particular not `shutdown`.

    Piece 0 yields no audio (`chunks=0`) but one non-pad frame, so priming runs straight into
    the anchor decision, and the held worker keeps the sizing queued while it waits."""
    runtime = _fake_runtime(chunks=0, frames=[_frame(5)])
    env = _env(envs, runtime, split_chars=100)
    held = _SizingHeldInQueue(monkeypatch)
    deciding = threading.Event()
    real_anchor_codes = routes_speech.anchor_codes

    def spy(*args: Any) -> Any:
        deciding.set()  # the last step before the GPU thread waits on the sizing
        return real_anchor_codes(*args)

    monkeypatch.setattr(routes_speech, "anchor_codes", spy)

    async def scenario() -> None:
        task = _serve(_text_request(SENTENCES), runtime, env.components)
        assert await asyncio.to_thread(deciding.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await eventually(lambda: _components_gate_is_free(env.components))

    try:
        asyncio.run(scenario())
    finally:
        held.release()

    [future] = held.futures
    assert future.cancelled()
    assert _events(env.events, "speech.anchor_skipped") == []
    assert held.sized == []


def test_a_sizing_failure_skips_the_anchor_instead_of_aborting_the_stream(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch, prepared: list[dict[str, Any]]
) -> None:
    """Finding 2: the sizing can fail for reasons of its own (a template or tokenizer error,
    or no CPU tokenizer installed). Piece 0 is already out by then, so the request degrades:
    the anchor is skipped (`sizing_failed`), and every later piece is generated as voice design.

    The error is still a server bug, reported at level error with its traceback, but as
    `speech.anchor_sizing_failed`, not `request.failed` (review 34 finding 4): the request goes
    on to a `200` and `speech.completed`, so a `request.failed` would count a request that
    succeeded as a failed one."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)

    def broken_sizing(*_args: Any) -> Any:
        raise ValueError("the template broke")

    monkeypatch.setattr(routes_speech, "anchor_sizing", broken_sizing)

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "sizing_failed"
    assert skipped["request_id"] == response.headers["x-request-id"]
    pieces = _no_reference_pieces(SENTENCES, 100)
    assert [p["text"] for p in prepared] == pieces
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    [failed] = _events(env.events, "speech.anchor_sizing_failed")
    assert failed["level"] == "error"
    assert failed["request_id"] == response.headers["x-request-id"]
    assert "the template broke" in str(failed["error"])
    assert "broken_sizing" in str(failed["traceback"])  # where it was raised, not just what
    assert _events(env.events, "request.failed") == []
    assert len(_events(env.events, "speech.completed")) == 1


def test_a_sizing_that_outlasts_its_timeout_skips_the_anchor(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch, prepared: list[dict[str, Any]]
) -> None:
    """Finding 3: the GPU thread waits for the sizing at most `ANCHOR_SIZING_TIMEOUT_SECONDS`
    (shortened here). A sizing still running past that is abandoned (`sizing_timeout`), and
    the stream goes on as voice design instead of holding the GPU thread -- and, behind it, a
    disconnect's `gen.close()` -- for as long as the CPU takes."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    finish_sizing = threading.Event()
    real_sizing = routes_speech.anchor_sizing

    def slow_sizing(*args: Any) -> Any:
        finish_sizing.wait(5)
        return real_sizing(*args)

    monkeypatch.setattr(routes_speech, "anchor_sizing", slow_sizing)
    monkeypatch.setattr(routes_speech, "ANCHOR_SIZING_TIMEOUT_SECONDS", 0.05)
    try:
        response = env.speak(text=SENTENCES)
        assert not finish_sizing.is_set()  # the stream finished without the sizing
    finally:
        finish_sizing.set()

    assert response.status_code == 200
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "sizing_timeout"
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    assert len(_events(env.events, "speech.completed")) == 1


def test_the_sizing_timeout_is_well_under_the_gpu_close_timeout() -> None:
    """Finding 3: a disconnect's `gen.close()` queues behind a step blocked on the sizing, and
    a close past `GPU_CLOSE_TIMEOUT_SECONDS` poisons the gate. The wait must leave the close
    most of that budget."""
    assert 0 < limits.ANCHOR_SIZING_TIMEOUT_SECONDS <= GPU_CLOSE_TIMEOUT_SECONDS / 4


def test_a_sizing_cancelled_by_shutdown_is_reported_as_shutdown(
    envs: list[Env], monkeypatch: pytest.MonkeyPatch, prepared: list[dict[str, Any]]
) -> None:
    """Finding 4: `CpuTokenizer.shutdown()` cancels a sizing still queued. The anchor is
    skipped for that reason (`shutdown`), not reported as `no_room`, which it wasn't."""
    runtime = _fake_runtime(chunks=3, frames=[_frame(5), _frame(6), _frame(7)])
    env = _env(envs, runtime, split_chars=100)
    held = _SizingHeldInQueue(monkeypatch)
    queue_sizing = routes_speech._start_anchor_sizing

    def start_then_shut_down(runtime: Any, cpu_tokenizer: Any, request: Any, pieces: Any) -> Any:
        future = queue_sizing(runtime, cpu_tokenizer, request, pieces)
        cpu_tokenizer.shutdown()  # cancels the sizing, still queued behind the held worker
        return future

    monkeypatch.setattr(routes_speech, "_start_anchor_sizing", start_then_shut_down)
    try:
        response = env.speak(text=SENTENCES)
    finally:
        held.release()

    assert response.status_code == 200
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "shutdown"
    assert all(isinstance(p["reference"], NoRef) for p in prepared)
    assert held.sized == []


# --- CpuTokenizer shutdown races (review #3 on 2d9070a) ---------------------------------------
#
# Direct unit tests against `routes_speech.CpuTokenizer` itself, not through the HTTP route:
# `api._drain_gpu` cancels every request it already knows about before calling `shutdown()`, so
# reaching this race through a live request would need timing this precise anyway.


def test_cpu_tokenizer_run_after_shutdown_raises_gpu_unavailable() -> None:
    """A call arriving after `shutdown()` hits `ThreadPoolExecutor`'s own `RuntimeError`
    ("cannot schedule new futures after shutdown"), mapped to the same `GpuUnavailable` a
    poisoned gate answers with (`503 gpu_unavailable`) -- not left to surface as a `500`."""
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(object(), object())
    tokenizer.shutdown()

    with pytest.raises(GpuUnavailable):
        asyncio.run(tokenizer.run(lambda _tokenizer: None))


def test_cpu_tokenizer_submit_after_shutdown_raises_gpu_unavailable() -> None:
    """`submit` (the GPU thread's own entry point, `_start_anchor_sizing`) maps the same
    executor `RuntimeError` the same way, even though nothing here is `await`ed."""
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(object(), object())
    tokenizer.shutdown()

    with pytest.raises(GpuUnavailable):
        tokenizer.submit(lambda _tokenizer: None)


def test_cpu_tokenizer_run_queued_call_cancelled_by_shutdown_raises_gpu_unavailable() -> None:
    """A call still queued -- behind another one already running, since the executor has one
    worker -- when `shutdown(cancel_futures=True)` runs is cancelled. Told apart from a real
    cancellation of the awaiting task by its own cancel count staying at zero (confirmed
    directly below), it is mapped to the same `GpuUnavailable`, not left to surface as a bare
    `asyncio.CancelledError` indistinguishable from a client disconnect."""
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(object(), object())
    running = threading.Event()
    release = threading.Event()

    def block(_tokenizer: Any) -> None:
        running.set()
        release.wait(timeout=5)

    async def scenario() -> None:
        first = asyncio.ensure_future(tokenizer.run(block))
        await asyncio.to_thread(running.wait, 5)
        task = asyncio.current_task()
        assert task is not None
        cancelling_before = task.cancelling()
        # The executor's one worker is now busy with `first`; this second call queues.
        second = asyncio.ensure_future(tokenizer.run(lambda _tokenizer: None))
        await asyncio.sleep(0)  # let `second` reach `self._executor.submit(...)` and queue
        tokenizer.shutdown()  # cancels `second` while it is still queued, not yet running
        release.set()
        await first
        with pytest.raises(GpuUnavailable):
            await second
        assert task.cancelling() == cancelling_before  # never itself cancelled

    asyncio.run(scenario())


def test_cpu_tokenizer_join_waits_for_both_workers_within_its_timeout() -> None:
    """Review 34 finding 6: a test teardown shuts the executors down, then `join`s them with a
    bound, so a call still running can't reach into the next test (which reuses the same
    session-shared tokenizer copy) and a wedged one fails the teardown instead of hanging it.
    It waits for a running call on each worker, the pre-gate one and the sizing one."""
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(FakeTokenizer(), FakeTokenizer())
    running = threading.Barrier(3)
    finished: list[str] = []

    def slow(name: str) -> Any:
        def call(_tokenizer: Any) -> None:
            running.wait(5)
            time.sleep(0.1)
            finished.append(name)

        return call

    pre_gate = threading.Thread(target=asyncio.run, args=(tokenizer.run(slow("pre_gate")),))
    pre_gate.start()
    tokenizer.submit(slow("sizing"))
    running.wait(5)
    tokenizer.shutdown()

    assert tokenizer.join(5)
    assert sorted(finished) == ["pre_gate", "sizing"]
    pre_gate.join(5)


def test_cpu_tokenizer_join_gives_up_on_a_wedged_call() -> None:
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(FakeTokenizer(), FakeTokenizer())
    running = threading.Event()
    release = threading.Event()

    def wedged(_tokenizer: Any) -> None:
        running.set()
        release.wait(5)

    tokenizer.submit(wedged)
    assert running.wait(5)
    tokenizer.shutdown()
    try:
        assert tokenizer.join(0.05) is False
    finally:
        release.set()
    assert tokenizer.join(5)


# --- the lease holder's sizing has its own worker (review 34 finding 5) -----------------------


def test_a_burst_of_pre_gate_checks_does_not_delay_the_lease_holders_sizing() -> None:
    """Every request's pre-gate room check queues on one worker, including ones bound for
    `409`. The lease holder's sizing has a worker of its own, so a burst of slow checks ahead
    of it can't push it past `ANCHOR_SIZING_TIMEOUT_SECONDS` and change the speaker mid-stream."""
    tokenizer = routes_speech.CpuTokenizer()
    tokenizer.install(FakeTokenizer(), FakeTokenizer())
    release = threading.Event()
    started = threading.Event()

    def slow_pre_gate_check(_tokenizer: Any) -> None:
        started.set()
        release.wait(5)

    async def scenario() -> None:
        burst = [asyncio.ensure_future(tokenizer.run(slow_pre_gate_check)) for _ in range(8)]
        try:
            assert await asyncio.to_thread(started.wait, 5)
            sizing = tokenizer.submit(lambda _tokenizer: "sized")
            assert await asyncio.to_thread(sizing.result, 1.0) == "sized"
            assert not any(check.done() for check in burst)  # sized while all still waited
        finally:
            release.set()
            await asyncio.gather(*burst)

    try:
        asyncio.run(scenario())
    finally:
        tokenizer.shutdown()
