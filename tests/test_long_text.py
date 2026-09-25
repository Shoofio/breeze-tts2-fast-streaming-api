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

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import torch

from breeze_infer import routes_speech
from breeze_infer.api import Components, create_app
from breeze_infer.gpu import GpuGate
from breeze_infer.limits import ANCHOR_CHARS
from breeze_infer.routes_health import Readiness
from breeze_infer.synthesis import CodesRef, NoRef, anchor_codes, prepare_piece
from breeze_infer.text_split import split_text
from models.fast_streaming import MIN_SUFFIX_FRAMES
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
from tests.test_speech_abort import LiveServer, wait_until

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


def test_partial_room_clamps_and_emits_piece_clamped(envs: list[Env]) -> None:
    """FR-036a: piece 1 has room to start but less than its cap. It is generated up to the
    room, ends normally, and the server records `speech.piece_clamped` with the client's
    `requested`, the server's `cap` and the `room`."""
    frames = [_frame(n) for n in range(1, 41)]
    pieces = _no_reference_pieces(SENTENCES, 100)
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    room = 30
    assert room >= MIN_SUFFIX_FRAMES  # enough that the anchor is kept (#4)
    max_seq_len = _seq_len(anchor, pieces[1]) + 1 + room
    cap = 50
    # Piece 0's prompt is much shorter, so it keeps its full cap; only later pieces clamp.
    assert max_seq_len - _seq_len(NoRef(), pieces[0]) - 1 >= cap
    runtime = _fake_runtime(
        chunks=40, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES, max_new_tokens=str(cap))

    assert response.status_code == 200
    clamped = _events(env.events, "speech.piece_clamped")
    # Every later piece carries the same anchor and a same-length sentence, so each is
    # clamped alike; piece 0 is not.
    assert [(e["piece_index"], e["requested"], e["cap"], e["room"]) for e in clamped] == [
        (index, cap, cap, room) for index in range(1, len(pieces))
    ]
    assert clamped[0]["request_id"] == response.headers["x-request-id"]
    assert runtime.calls[0]["max_new_tokens"] == cap
    assert runtime.calls[1]["max_new_tokens"] == room
    assert runtime.calls[1]["inputs"]["input_values"] is not None  # anchored
    done = {e["piece_index"]: e["frames"] for e in _events(env.events, "speech.piece_done")}
    assert done[0] == 40
    assert done[1] == room
    assert len(_events(env.events, "speech.completed")) == 1


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


def test_an_anchor_that_leaves_a_later_piece_too_little_room_is_skipped(
    envs: list[Env], prepared: list[dict[str, Any]]
) -> None:
    """With piece 0 as its reference, piece 1 would have less than MIN_SUFFIX_FRAMES left. The
    anchor is skipped whole (never trimmed: its codes must match its text), and the later
    pieces are voice design, which fits."""
    frames = [_frame(n) for n in range(1, 5)]
    pieces = _no_reference_pieces(SENTENCES, 100)
    anchor = CodesRef(codes=torch.stack(frames), ref_text=pieces[0])
    max_seq_len = _seq_len(anchor, pieces[1]) + 1 + (MIN_SUFFIX_FRAMES - 1)
    runtime = _fake_runtime(
        chunks=4, frames=frames, config=FakeStreamingConfig(max_seq_len=max_seq_len)
    )
    env = _env(envs, runtime, split_chars=100)

    response = env.speak(text=SENTENCES)

    assert response.status_code == 200
    # The anchor was tried for piece 1 first; each piece's last preparation is what ran.
    used = {p["text"]: p["reference"] for p in prepared}
    assert list(used) == pieces
    assert all(isinstance(reference, NoRef) for reference in used.values())
    [skipped] = _events(env.events, "speech.anchor_skipped")
    assert skipped["reason"] == "no_room"
    assert len(_events(env.events, "speech.completed")) == 1


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
