"""Tests for the pure WebSocket session state machine (breeze_infer/ws_session.py, T072/T076).

The session holds no I/O: `apply` turns one parsed client message into its immediate events, and
a worker drains the work deque. `FakeWorker` below stands in for ws_server.py's worker coroutine,
so ordering and exactly-once rules (BC-34..BC-40, SC-005) are checked without sockets or a GPU.

Interface these tests pin (T076 implements to it):

- Client messages are built only with `ws_messages.parse(json_str)`, never by constructing them.
- `Session(*, lookup_voice, default_split_chars)`: `lookup_voice(voice_id)` returns the voice or
  `None` (VoiceRegistry.lookup's shape); `default_split_chars` is the launch setting used when
  `start` omits `split_chars` (parse has no settings, so it can't fill that default in).
- `session.apply(message) -> list[dict]` of immediate events, in wire shape:
  `{"type": "instruction_set"}` and
  `{"type": "error", "code": str, "message": str, "request_type": str | None}`.
  A valid `start`, `text`, `flush`, `end` or `cancel` returns `[]`: `started`, `cancelled` and
  `done` come from the worker, through the work deque, so they keep message order.
- `session.next_item()` returns, in order, `Piece(epoch, index, text)`, `EndMark(epoch)`, a
  `CancelMark` instance, `StartMark(config)`, or `None` when the deque is empty.
- `session.is_stale(piece)` takes a `Piece`: true once its epoch is not current. An `EndMark`
  needs no check: a cancel that supersedes one removes it from the deque, so the worker sends
  `done` for every EndMark it gets.
- `session.mark_piece_done(anchor)`: the worker calls it exactly once for every `Piece` that
  `next_item` returned, when it stops working on it (finished, failed, cancelled, or skipped as
  stale). `anchor` is the reference built from a piece that succeeded, or `None` when it failed
  or was cut short. The session tracks the piece in flight itself (from `next_item`), keeps the
  first anchor from a current (non-stale) piece, and ignores the rest. A current piece ending with
  `None` and no anchor yet counts as a failure (data-model "States").
- `session.config.voice_id` (`""` when none; `StartMark.config` has it too, for `started`),
  `session.config.instruction` (read by the worker when each piece starts), `session.anchor`.
- `session.close()` on disconnect: the piece in flight goes stale and the deque is emptied, with
  no `CancelMark`, so nothing more is sent; `apply` is not called again.

Session rules pinned here (data-model.md "WebSocket Session", States):
- "Pending work" for `start` (BC-36) is a `Piece` queued or in flight. A queued `EndMark` alone
  still gets its `done` before `started`; buffered text alone is dropped silently.
- A valid `start` always resets buffer, anchor, opening flag and `piece_index`.
- `cancel` keeps the anchor and `piece_index`; it reopens the opening budget only when piece 0 is
  cancelled before it anchored. A `cancel` before `start` is `not_started`.
- The first piece that succeeds provides the anchor; if piece 0 fails, the opening budget turns
  back on only when no piece is queued.
- A rejected message has no other effect: a `flush`/`end` hitting `text_too_long` neither drains
  nor queues its `EndMark`.
"""

from __future__ import annotations

import json
import random
import re

import pytest

from breeze_infer import ws_messages, ws_session
from breeze_infer.http_fields import DEFAULT_INSTRUCTION
from breeze_infer.limits import ANCHOR_CHARS, MAX_TEXT_CHARS
from breeze_infer.text_split import segment, weigh

# Imported through the package rather than `from breeze_infer.ws_session import ...`: until
# T076 creates the module, ruff's import sorter can't tell it is first-party.
Session = ws_session.Session
Piece = ws_session.Piece
EndMark = ws_session.EndMark
CancelMark = ws_session.CancelMark
StartMark = ws_session.StartMark

ALICE = object()  # a stand-in resolved voice: the session only needs to know one exists
VOICES = {"alice": ALICE}


def msg(kind: str, **fields: object) -> object:
    """One client message, through the real parser."""
    parsed = ws_messages.parse(json.dumps({"type": kind, **fields}))
    assert not isinstance(parsed, ws_messages.WsError), parsed
    return parsed


def new_session(split_chars: int = 600) -> Session:
    return Session(lookup_voice=VOICES.get, default_split_chars=split_chars)


def started_session(split_chars: int = 600, **start_fields: object) -> tuple[Session, FakeWorker]:
    """A session past `start`, with the worker having sent `started`."""
    session = new_session(split_chars)
    assert session.apply(msg("start", **start_fields)) == []
    worker = FakeWorker(session)
    worker.run()
    assert worker.kinds() == ["started"]
    worker.events.clear()
    return session, worker


def error(code: str, message: str, request_type: str | None) -> dict:
    return {"type": "error", "code": code, "message": message, "request_type": request_type}


def drain_pieces(session: Session) -> list[Piece]:
    """Everything queued, asserting it is all pieces."""
    items = []
    while (item := session.next_item()) is not None:
        assert isinstance(item, Piece), item
        items.append(item)
    return items


class FakeWorker:
    """ws_server.py's worker loop without the GPU: one item at a time, a piece is `chunks` audio
    frames, and the piece in flight is checked for staleness between frames (R15).

    `step()` does one unit of work: take an item (emitting its marker, or `speaking` for a
    piece), or emit one audio frame of the piece in flight. Events are wire-shaped dicts plus
    bookkeeping keys starting with `_`: `_at` is the number of client messages applied when the
    event was emitted (`now`, set by the driver), `_aborted` marks a piece cut short, `_index`
    is the piece index, and `_ended_at` is `now` when the piece ended however it ended. Pieces
    whose index is in `fail` fail before their first frame, as a
    generation error does: `error{generation_failed}`, then `mark_piece_done(None)`.
    """

    def __init__(self, session: Session, chunks: int = 2) -> None:
        self.session = session
        self.chunks = chunks
        self.events: list[dict] = []
        self.now = 0
        self._current: Piece | None = None
        self._speaking: dict | None = None
        self._left = 0
        self.fail: set[int] = set()

    def _emit(self, event: dict) -> dict:
        event["_at"] = self.now
        self.events.append(event)
        return event

    def _finish(self, anchor: object) -> None:
        self._speaking["_ended_at"] = self.now
        self._current = None
        self.session.mark_piece_done(anchor)

    def step(self) -> bool:
        """One unit of work; False when there was nothing to do."""
        if self._current is not None:
            if self.session.is_stale(self._current):
                self._speaking["_aborted"] = True
                self._finish(None)
                return True
            if self._current.index in self.fail and self._left == self.chunks:
                self._emit({"type": "error", "code": "generation_failed"})
                self._finish(None)
                return True
            self._emit({"type": "audio", "index": self._current.index})
            self._left -= 1
            if self._left == 0:
                self._finish(("anchor", self._current.epoch, self._current.index))
            return True

        item = self.session.next_item()
        if item is None:
            return False
        if isinstance(item, CancelMark):
            self._emit({"type": "cancelled"})
        elif isinstance(item, StartMark):
            self._emit({"type": "started", "voice_id": item.config.voice_id})
        elif isinstance(item, EndMark):
            self._emit({"type": "done"})
        elif isinstance(item, Piece):
            if self.session.is_stale(item):
                self.session.mark_piece_done(None)
            else:
                self._current = item
                self._left = self.chunks
                self._speaking = self._emit(
                    {"type": "speaking", "text": item.text, "_aborted": False, "_index": item.index}
                )
        else:
            pytest.fail(f"unexpected work item {item!r}")
        return True

    def run(self) -> None:
        for _ in range(100_000):
            if not self.step():
                return
        raise AssertionError("worker never went idle")

    def kinds(self, *, audio: bool = False) -> list[str]:
        return [e["type"] for e in self.events if audio or e["type"] != "audio"]

    def spoken(self) -> list[str]:
        return [e["text"] for e in self.events if e["type"] == "speaking"]


# ------------------------------------------------------------------------------ done / cancel


def test_bc_34_end_with_nothing_left_still_sends_one_done() -> None:
    """C++ sent no `done` when `end` arrived with nothing left to speak, so clients hung."""
    session, worker = started_session()
    assert session.apply(msg("end")) == []
    worker.run()
    assert worker.kinds() == ["done"]

    # Everything already spoken by `flush`: the `end` after it still gets its own `done`.
    session.apply(msg("text", text="Hello."))
    session.apply(msg("flush"))
    worker.run()
    session.apply(msg("end"))
    session.apply(msg("end"))
    worker.run()
    assert worker.kinds() == ["done", "speaking", "done", "done"]
    assert worker.spoken() == ["Hello."]


def test_end_done_comes_after_every_piece_queued_before_it() -> None:
    session, worker = started_session()
    session.apply(msg("text", text="One. Two. Thr"))
    session.apply(msg("end", text="ee"))
    session.apply(msg("text", text="Four. Five"))
    worker.run()
    # "Four." was sent after `end`, so it is spoken after the `done`.
    assert worker.kinds() == ["speaking", "speaking", "done", "speaking"]
    assert worker.spoken() == ["One. Two.", "Three", "Four."]


def test_bc_35_every_cancel_gets_exactly_one_cancelled_even_when_idle() -> None:
    """C++ latched a cancel flag: an idle `cancel` went unacknowledged, or its `cancelled` came
    later as a spurious reply to other work."""
    session, worker = started_session()
    assert session.apply(msg("cancel")) == []
    worker.run()
    assert worker.kinds() == ["cancelled"]

    for _ in range(3):
        assert session.apply(msg("cancel")) == []
    worker.run()
    assert worker.kinds() == ["cancelled"] * 4

    # And with work in flight: still exactly one each.
    session.apply(msg("text", text="One. Two. Three. X"))
    worker.step()  # speaking "One. Two. Three." or its first piece
    session.apply(msg("cancel"))
    session.apply(msg("cancel"))
    worker.run()
    assert worker.kinds().count("cancelled") == 6


def test_bc_35_cancel_between_pieces_never_drops_later_pieces() -> None:
    """C++'s latched cancel flag could swallow the next piece sent after a `cancel`."""
    session, worker = started_session()
    session.apply(msg("text", text="One. Tail"))
    worker.run()
    assert worker.spoken() == ["One."]

    # Idle between pieces: cancel, then new text is spoken in full.
    session.apply(msg("cancel"))
    worker.run()
    session.apply(msg("text", text="Two. Three. X"))
    session.apply(msg("flush"))
    worker.run()
    assert worker.kinds() == ["speaking", "cancelled", "speaking", "speaking"]
    assert worker.spoken() == ["One.", "Two. Three.", "X"]

    # A piece in flight: cancel, then text arrives before the worker notices the cancel. The
    # new piece belongs to the new epoch and is spoken after `cancelled`.
    worker.events.clear()
    session.apply(msg("text", text="Four. Tail"))
    worker.step()  # speaking "Four."
    in_flight = worker._current
    session.apply(msg("cancel"))
    assert session.is_stale(in_flight)
    session.apply(msg("text", text="Five. More"))
    worker.run()
    assert worker.kinds() == ["speaking", "cancelled", "speaking"]
    assert worker.spoken() == ["Four.", "Five."]
    speaking = [e for e in worker.events if e["type"] == "speaking"]
    assert [e["_aborted"] for e in speaking] == [True, False]
    # "Tail" was discarded by the cancel; "More" is still buffered.
    session.apply(msg("flush"))
    worker.run()
    assert worker.spoken()[-1] == "More"


def test_cancel_discards_buffered_text() -> None:
    session, worker = started_session()
    session.apply(msg("text", text="Unfinished sentence"))
    session.apply(msg("cancel"))
    session.apply(msg("flush"))
    worker.run()
    assert worker.kinds() == ["cancelled"]


@pytest.mark.parametrize("stage", ["queued", "in_flight", "pieces_spoken"])
def test_a_later_cancel_replaces_a_pending_done(stage: str) -> None:
    session, worker = started_session()
    worker.chunks = 1
    session.apply(msg("end", text="One. Two."))
    if stage == "in_flight":
        worker.step()
        assert worker.kinds(audio=True) == ["speaking"]
    elif stage == "pieces_spoken":
        # "One. Two." fits one piece: speaking, its one frame, and only the EndMark is left.
        worker.step()
        worker.step()
        assert worker.kinds(audio=True) == ["speaking", "audio"]
        assert worker._current is None
    session.apply(msg("cancel"))
    worker.run()
    kinds = worker.kinds()
    assert "done" not in kinds
    assert kinds.count("cancelled") == 1
    assert kinds[-1] == "cancelled"


# ------------------------------------------------------------------------------ start


def test_start_on_an_idle_session_sends_only_started() -> None:
    session = new_session()
    worker = FakeWorker(session)
    assert session.apply(msg("start")) == []
    worker.run()
    assert worker.events == [{"type": "started", "voice_id": "", "_at": 0}]

    # Idle again after work: a second start is only `started`, and piece indices restart.
    session.apply(msg("text", text="One. Tail"))
    session.apply(msg("text", text=" Two. X"))
    session.apply(msg("flush"))
    first = []
    for _ in range(3):
        first.append(session.next_item())
        session.mark_piece_done(None)
    assert [(p.index, p.text) for p in first] == [(0, "One."), (1, "Tail Two."), (2, "X")]
    worker.events.clear()
    assert session.apply(msg("start", voice_id="alice")) == []
    worker.run()
    assert worker.kinds() == ["started"]
    assert worker.events[0]["voice_id"] == "alice"
    assert session.config.voice_id == "alice"

    session.apply(msg("flush", text="Again."))
    assert [(p.index, p.text) for p in drain_pieces(session)] == [(0, "Again.")]


@pytest.mark.parametrize("in_flight", [False, True])
def test_bc_36_start_with_pending_work_cancels_then_starts(in_flight: bool) -> None:
    """C++ applied a mid-speech `start` to the running session in place (a data race), with no
    `cancelled` for the interrupted work."""
    session, worker = started_session()
    session.apply(msg("text", text="One. Two. Three. Tail"))
    if in_flight:
        worker.step()
        piece = worker._current
        assert piece is not None
    assert session.apply(msg("start", voice_id="alice")) == []
    if in_flight:
        assert session.is_stale(piece)
    worker.run()

    expected = ["speaking", "cancelled", "started"] if in_flight else ["cancelled", "started"]
    assert worker.kinds() == expected
    assert worker.events[-1]["voice_id"] == "alice"
    if in_flight:
        assert worker.events[0]["_aborted"] is True

    # The interrupted session's buffer went with it; the new one starts at piece 0.
    session.apply(msg("flush", text="Fresh."))
    items = drain_pieces(session)
    assert [(p.index, p.text) for p in items] == [(0, "Fresh.")]


def test_start_after_only_an_end_mark_sends_done_then_started() -> None:
    session, worker = started_session()
    worker.chunks = 1
    session.apply(msg("end", text="Hi."))
    worker.step()
    worker.step()  # "Hi." spoken in full; only the EndMark is left
    assert worker.kinds(audio=True) == ["speaking", "audio"]
    session.apply(msg("start", voice_id="alice"))
    worker.run()
    assert worker.kinds() == ["speaking", "done", "started"]


def test_cancel_in_a_new_session_keeps_the_previous_sessions_done() -> None:
    """Review 46 #3: a start that interrupts nothing leaves the old session's EndMark queued,
    and a cancel in the new session supersedes only the new session's work."""
    session, worker = started_session()
    session.apply(msg("end"))  # EndMark queued, no pieces
    session.apply(msg("start", voice_id="alice"))  # nothing pending: no cancel
    session.apply(msg("cancel"))
    worker.run()
    assert worker.kinds() == ["done", "started", "cancelled"]

    # Same with pieces queued in the new session: they go, the old done stays.
    session.apply(msg("end"))
    session.apply(msg("start"))
    session.apply(msg("text", text="One. Two. Tail"))
    session.apply(msg("cancel"))
    worker.run()
    assert worker.kinds()[3:] == ["done", "started", "cancelled"]


def test_close_empties_the_queue_including_an_earlier_sessions_done() -> None:
    """Review 47 #4: after a disconnect nothing may be sent, not even the `done` an earlier
    session left queued (a cancel would keep that one)."""
    session, worker = started_session()
    session.apply(msg("end"))
    session.apply(msg("start"))  # interrupts nothing: the EndMark stays queued
    session.apply(msg("text", text="One. Two. Tail"))
    session.close()
    assert session.next_item() is None
    worker.run()
    assert worker.events == []


def test_close_stops_the_piece_in_flight_without_a_cancelled() -> None:
    session, worker = started_session()
    session.apply(msg("end", text="One. Two."))
    worker.step()
    piece = worker._current
    assert piece is not None
    session.close()
    assert session.is_stale(piece)
    assert session.next_item() is None
    # The piece ends; its anchor is ignored, as for any stale piece.
    session.mark_piece_done(("anchor", piece.epoch, piece.index))
    worker._current = None
    assert session.anchor is None
    assert worker.kinds() == ["speaking"]


def test_start_drops_buffered_text_silently() -> None:
    session, worker = started_session()
    session.apply(msg("text", text="Unfinished sentence"))
    session.apply(msg("start"))
    session.apply(msg("flush"))
    worker.run()
    assert worker.kinds() == ["started"]


def test_null_fields_on_start_count_as_absent() -> None:
    session = new_session(split_chars=0)
    worker = FakeWorker(session)
    session.apply(msg("start", voice_id=None, instruction=None, split_chars=None))
    worker.run()
    assert worker.events[0]["voice_id"] == ""
    assert session.config.instruction == DEFAULT_INSTRUCTION
    # The launch default (0 here: no length splitting) applies.
    session.apply(msg("flush", text=LONG_SENTENCES))
    assert [p.text for p in drain_pieces(session)] == [LONG_SENTENCES]


def test_piece_index_keeps_counting_after_cancel_and_resets_at_start() -> None:
    session, _ = started_session()
    session.apply(msg("text", text="One. Tail"))
    session.apply(msg("cancel"))  # piece 0 dropped while queued
    session.apply(msg("text", text="Two. More"))
    assert isinstance(session.next_item(), CancelMark)
    piece = session.next_item()
    assert (piece.index, piece.text) == (1, "Two.")

    # Cancelled in flight: still counted.
    session.apply(msg("cancel"))
    session.apply(msg("text", text=" Three. X"))
    session.mark_piece_done(None)
    assert isinstance(session.next_item(), CancelMark)
    piece = session.next_item()
    assert (piece.index, piece.text) == (2, "Three.")
    session.mark_piece_done(("anchor", piece.epoch, piece.index))

    session.apply(msg("start"))
    session.apply(msg("flush", text="Four."))
    assert isinstance(session.next_item(), StartMark)
    piece = session.next_item()
    assert (piece.index, piece.text) == (0, "Four.")


def test_unknown_voice_id_on_start_leaves_the_previous_session_untouched() -> None:
    session, worker = started_session(instruction="Custom style.")
    session.apply(msg("text", text="Hello there. Wor"))
    out = session.apply(msg("start", voice_id="nobody", instruction="Other."))
    assert out == [error("unknown_voice", "unknown voice_id", "start")]
    assert session.config.voice_id == ""
    assert session.config.instruction == "Custom style."

    worker.run()
    assert worker.kinds() == ["speaking"]  # no `cancelled`, no second `started`
    session.apply(msg("flush", text="ld"))
    worker.run()
    assert worker.spoken() == ["Hello there.", "World"]


def test_unknown_voice_id_on_the_first_start_leaves_the_session_not_started() -> None:
    session = new_session()
    assert session.apply(msg("start", voice_id="nobody")) == [
        error("unknown_voice", "unknown voice_id", "start")
    ]
    assert session.next_item() is None
    assert session.apply(msg("text", text="Hi.")) == [
        error("not_started", "send start first", "text")
    ]


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("text", {"text": "Hello."}),
        ("flush", {}),
        ("end", {}),
        ("instruction", {"instruction": "Whisper."}),
        ("cancel", {}),
    ],
)
def test_not_started_before_start(kind: str, fields: dict) -> None:
    session = new_session()
    assert session.apply(msg(kind, **fields)) == [
        error("not_started", "send start first", kind)
    ]
    assert session.next_item() is None
    # Nothing was buffered or queued either.
    session.apply(msg("start"))
    session.apply(msg("end"))
    worker = FakeWorker(session)
    worker.run()
    assert worker.kinds() == ["started", "done"]


# ------------------------------------------------------------------------------ instruction


@pytest.mark.parametrize("blank", ["", "   "])
def test_bc_37_blank_instruction_resets_to_default(blank: str) -> None:
    """C++ stored an empty `instruction` message as `""`, so later pieces had no instruction."""
    session, _ = started_session(instruction="Whisper softly.")
    assert session.config.instruction == "Whisper softly."

    assert session.apply(msg("instruction", instruction="Shout.")) == [{"type": "instruction_set"}]
    assert session.config.instruction == "Shout."

    assert session.apply(msg("instruction", instruction=blank)) == [{"type": "instruction_set"}]
    assert session.config.instruction == DEFAULT_INSTRUCTION


def test_start_without_instruction_uses_the_default() -> None:
    session, _ = started_session()
    assert session.config.instruction == DEFAULT_INSTRUCTION


# ------------------------------------------------------------------------------ segmenting

LONG_SENTENCES = " ".join(f"Sentence number {i} has quite a few words in it." for i in range(12))


def test_bc_39_end_of_buffer_punctuation_waits() -> None:
    """C++ cut at a period at the very end of the buffer, so `3.` then `14` became two pieces."""
    session, worker = started_session()
    session.apply(msg("text", text="It costs 3."))
    assert session.next_item() is None
    session.apply(msg("text", text="14 today. Next"))
    worker.run()
    assert worker.spoken() == ["It costs 3.14 today."]

    # A CJK stop at the end waits for a closer that may follow.
    session.apply(msg("text", text=" words.\n「你好。"))
    worker.run()
    assert worker.spoken() == ["It costs 3.14 today.", "Next words."]
    session.apply(msg("text", text="」好"))
    worker.run()
    assert worker.spoken()[-1] == "「你好。」"

    # `flush` speaks the waiting text.
    session.apply(msg("text", text="的。"))
    session.apply(msg("flush"))
    worker.run()
    assert worker.spoken()[-1] == "好的。"


def test_opening_budget_applies_to_the_first_piece_only() -> None:
    session, _ = started_session()
    buffer = LONG_SENTENCES + " Tail"
    session.apply(msg("text", text=buffer))
    pieces = drain_pieces(session)
    expected, _ = segment(buffer, budget=600, first_budget=ANCHOR_CHARS, final=False)
    assert [p.text for p in pieces] == expected
    assert weigh(pieces[0].text) <= ANCHOR_CHARS
    assert any(weigh(p.text) > ANCHOR_CHARS for p in pieces[1:])

    # The next drain gets the full budget from its first piece on (C++ applied the opening
    # budget to every piece of whichever drain came first).
    session.apply(msg("text", text=" " + LONG_SENTENCES + " End"))
    later = drain_pieces(session)
    expected_later, _ = segment(
        "Tail " + LONG_SENTENCES + " End", budget=600, first_budget=0, final=False
    )
    assert [p.text for p in later] == expected_later
    assert weigh(later[0].text) > ANCHOR_CHARS


def test_no_opening_budget_with_a_reference_or_without_splitting() -> None:
    session, _ = started_session(voice_id="alice")
    session.apply(msg("text", text=LONG_SENTENCES + " Tail"))
    pieces = drain_pieces(session)
    assert weigh(pieces[0].text) > ANCHOR_CHARS

    session, _ = started_session(split_chars=0)
    session.apply(msg("text", text=LONG_SENTENCES + " Tail"))
    pieces = drain_pieces(session)
    assert [p.text for p in pieces] == [LONG_SENTENCES]


@pytest.mark.parametrize("in_flight", [False, True])
def test_opening_flag_resets_when_piece_0_is_cancelled(in_flight: bool) -> None:
    session, worker = started_session()
    session.apply(msg("text", text="Hello there. Tail"))
    piece_0 = None
    if in_flight:
        worker.step()
        piece_0 = worker._current
        assert piece_0.index == 0
    session.apply(msg("cancel"))
    session.apply(msg("text", text=LONG_SENTENCES + " More"))
    if in_flight:
        # The stale piece 0 reports an anchor anyway: the session must not keep it.
        session.mark_piece_done(("anchor", piece_0.epoch, 0))
        worker._current = None
        assert session.anchor is None
    worker.run()
    spoken = worker.spoken()[1:] if in_flight else worker.spoken()
    expected, _ = segment(LONG_SENTENCES + " More", budget=600, first_budget=ANCHOR_CHARS, final=False)
    assert spoken == expected
    assert weigh(spoken[0]) <= ANCHOR_CHARS


def test_opening_flag_stays_off_once_piece_0_has_anchored() -> None:
    session, worker = started_session()
    session.apply(msg("text", text="Hello there. Tail"))
    worker.run()
    anchor = session.anchor
    assert anchor is not None
    session.apply(msg("cancel"))
    session.apply(msg("text", text=LONG_SENTENCES + " More"))
    worker.run()
    assert session.anchor is anchor
    assert weigh(worker.spoken()[1]) > ANCHOR_CHARS


def test_first_successful_piece_anchors_when_piece_0_fails_with_more_queued() -> None:
    session, worker = started_session()
    worker.fail = {0}
    session.apply(msg("text", text="Hello there. Tail"))
    session.apply(msg("text", text=" again. Rest"))
    worker.step()
    worker.step()  # piece 0 fails; piece 1 is still queued
    assert worker.kinds() == ["speaking", "error"]
    assert session.anchor is None

    # Piece 1 is queued and can anchor, so the opening budget stays off.
    session.apply(msg("text", text=" " + LONG_SENTENCES + " More"))
    worker.run()
    assert session.anchor[2] == 1  # the anchor came from piece 1, the first to succeed
    spoken = worker.spoken()
    assert spoken[:2] == ["Hello there.", "Tail again."]
    assert weigh(spoken[2]) > ANCHOR_CHARS


def test_opening_budget_reopens_when_piece_0_fails_with_nothing_queued() -> None:
    session, worker = started_session()
    worker.fail = {0}
    session.apply(msg("text", text="Hello there. Tail"))
    worker.run()
    assert worker.kinds() == ["speaking", "error"]
    assert session.anchor is None

    session.apply(msg("text", text=" " + LONG_SENTENCES + " More"))
    later = drain_pieces(session)
    expected, _ = segment(
        " Tail " + LONG_SENTENCES + " More", budget=600, first_budget=ANCHOR_CHARS, final=False
    )
    assert [p.text for p in later] == expected
    assert weigh(later[0].text) <= ANCHOR_CHARS


def test_bc_40_buffer_over_limit_is_an_error_and_not_appended() -> None:
    """C++ buffered unpunctuated text without bound."""
    session, worker = started_session(split_chars=0)
    session.apply(msg("text", text="a" * (MAX_TEXT_CHARS - 10)))
    assert session.next_item() is None

    out = session.apply(msg("text", text="b" * 11))
    assert len(out) == 1
    assert out[0]["type"] == "error"
    assert out[0]["code"] == "text_too_long"
    assert out[0]["request_type"] == "text"

    # Exactly at the limit is accepted, and the rejected text was never appended.
    assert session.apply(msg("text", text="c" * 10)) == []
    session.apply(msg("flush"))
    worker.run()
    assert worker.spoken() == ["a" * (MAX_TEXT_CHARS - 10) + "c" * 10]


@pytest.mark.parametrize("kind", ["flush", "end"])
def test_text_too_long_on_flush_or_end_has_no_other_effect(kind: str) -> None:
    session, worker = started_session(split_chars=0)
    session.apply(msg("text", text="a" * (MAX_TEXT_CHARS - 10)))
    out = session.apply(msg(kind, text="b" * 11))
    assert [(e["code"], e["request_type"]) for e in out] == [("text_too_long", kind)]
    # Neither drained nor queued an EndMark: nothing to do, and no `done` ever comes.
    assert session.next_item() is None
    session.apply(msg("flush"))
    worker.run()
    assert worker.kinds() == ["speaking"]
    assert worker.spoken() == ["a" * (MAX_TEXT_CHARS - 10)]


# ------------------------------------------------------------------------------ SC-005

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "fox", "golf", "hotel"]
SEPARATORS = [" ", " ", " ", ", ", ". ", "! ", "? ", "; ", "\n"]
SEEDS = range(1_000)
# Frames `parse` rejects: the server replies with the error and never calls the session.
INVALID_FRAMES = [
    "not json",
    '{"type":"bogus"}',
    '{"type":"text","text":5}',
    '{"type":"start","seed":-1}',
    '{"type":"start","split_chars":-1}',
    '{"type":"end","text":"a\\u0000b"}',
]
OVERSIZED = "zulu " * (MAX_TEXT_CHARS // 5) + "z"  # over the limit whatever is buffered


def letters(text: str) -> str:
    """Only what a piece can't lose: pieces are stripped, split at spaces and dropped when they
    have no letter, so the letters in order are what "nothing lost" compares."""
    return re.sub(r"[^a-z]", "", text)


def random_text(rng: random.Random) -> str:
    text = "".join(rng.choice(WORDS) + rng.choice(SEPARATORS) for _ in range(rng.randint(1, 6)))
    return text.rstrip() if rng.random() < 0.3 else text


def random_start(rng: random.Random) -> dict:
    fields: dict = {}
    if rng.random() < 0.4:
        fields["voice_id"] = "alice"
    if rng.random() < 0.5:
        fields["split_chars"] = rng.choice([0, 12, 40, 600])
    return fields


def run_sequence(seed: int) -> tuple[list[tuple[str, str]], list[dict]]:
    """Apply a random sequence of messages with random worker progress (and the odd failed
    piece) in between.

    Returns the boundary-relevant messages as `(kind, letters of its text)` and the events.
    `instruction`, rejected `start`s, oversized text and frames `parse` rejects are not
    recorded: they emit no markers, and a rejected message must have no other effect, which the
    checker would see as unexplained markers or unsent letters ("zulu") being spoken.
    """
    rng = random.Random(seed)
    session = new_session(rng.choice([0, 12, 40, 600]))
    worker = FakeWorker(session, chunks=rng.randint(1, 3))
    worker.fail = {i for i in range(200) if rng.random() < 0.05}
    messages: list[tuple[str, str]] = []

    def send(kind: str, **fields: object) -> None:
        assert session.apply(msg(kind, **fields)) == [], (seed, kind)
        messages.append((kind, letters(str(fields.get("text", "")))))
        worker.now = len(messages)

    send("start", **random_start(rng))
    for _ in range(rng.randint(5, 40)):
        if rng.random() < 0.35:
            for _ in range(rng.randint(1, 4)):
                worker.step()
            continue
        kind = rng.choices(
            [
                "text", "flush", "end", "cancel", "start", "instruction", "bad_start",
                "oversized", "invalid",
            ],
            weights=[40, 8, 14, 14, 8, 10, 3, 3, 3],
        )[0]
        if kind == "text":
            send("text", text=random_text(rng))
        elif kind in ("flush", "end"):
            send(kind, **({"text": random_text(rng)} if rng.random() < 0.5 else {}))
        elif kind == "cancel":
            send("cancel")
        elif kind == "start":
            send("start", **random_start(rng))
        elif kind == "instruction":
            instruction = rng.choice(["", "Whisper.", "Shout."])
            out = session.apply(msg("instruction", instruction=instruction))
            assert out == [{"type": "instruction_set"}], seed
        elif kind == "bad_start":
            out = session.apply(msg("start", voice_id="nobody"))
            assert out == [error("unknown_voice", "unknown voice_id", "start")], seed
        elif kind == "oversized":
            request_type = rng.choice(["text", "flush", "end"])
            out = session.apply(msg(request_type, text=OVERSIZED))
            assert [(e["code"], e["request_type"]) for e in out] == [
                ("text_too_long", request_type)
            ], seed
        else:
            assert isinstance(ws_messages.parse(rng.choice(INVALID_FRAMES)), ws_messages.WsError)
    worker.run()
    return messages, worker.events


def check_sequence(seed: int, messages: list[tuple[str, str]], events: list[dict]) -> int:
    """Assert SC-005 and BC-36 over one sequence; returns how many starts interrupted work."""
    markers = [e for e in events if e["type"] in ("cancelled", "done", "started")]
    owner: dict[int, int] = {}  # id(marker event) -> index of the message that caused it
    interrupting: set[int] = set()  # starts that cancelled pending work
    done_for: dict[int, dict] = {}
    p = 0

    def take(kind: str, i: int) -> None:
        nonlocal p
        assert p < len(markers) and markers[p]["type"] == kind, (seed, i, kind, markers[p:p + 3])
        owner[id(markers[p])] = i
        p += 1

    # `cancelled`/`done`/`started` come strictly in the order of the messages that caused them.
    # Whether a `start` rightly sent a `cancelled` first (BC-36) is checked further down.
    for i, (kind, _) in enumerate(messages):
        if kind == "cancel":
            take("cancelled", i)
        elif kind == "start":
            if p < len(markers) and markers[p]["type"] == "cancelled":
                interrupting.add(i)
                take("cancelled", i)
            take("started", i)
        elif kind == "end" and p < len(markers) and markers[p]["type"] == "done":
            done_for[i] = markers[p]
            take("done", i)
    assert p == len(markers), (seed, "unexplained markers", markers[p:])

    # Each end gets its done unless a cancel (or interrupting start) in its own session
    # superseded it: only the first cancel/start after the end can, since a start that
    # interrupts nothing ends the session and leaves the end's done owed (review 46 #3). A done
    # is emitted before its superseder was applied.
    def supersedes(j: int | None) -> bool:
        return j is not None and (messages[j][0] == "cancel" or j in interrupting)

    def next_boundary(i: int) -> int | None:
        return next(
            (j for j in range(i + 1, len(messages)) if messages[j][0] in ("cancel", "start")), None
        )

    for i, (kind, _) in enumerate(messages):
        if kind != "end":
            continue
        j = next_boundary(i)
        if i in done_for:
            if supersedes(j):
                assert done_for[i]["_at"] <= j, (seed, "done after its superseder", i, j)
        else:
            assert supersedes(j), (seed, "end without done", i)

    # Segments: each cancel/start begins one. Its marker(s) in the event stream begin the same
    # segment there, since everything shares one ordered deque.
    segment_of: list[int] = []
    count = 0
    for kind, _ in messages:
        count += kind in ("cancel", "start")
        segment_of.append(count)
    sent: dict[int, str] = {}
    flushed: dict[int, str] = {}
    for i, (kind, text) in enumerate(messages):
        k = segment_of[i]
        sent[k] = sent.get(k, "") + text
        if kind in ("flush", "end"):
            flushed[k] = sent[k]

    spoken: dict[int, str] = {}
    speaking_in: dict[int, list[dict]] = {}
    aborted_in: set[int] = set()
    current = 0
    # Piece indices (and so seeds) count from 0 at each start and keep counting through
    # cancels: each speaking is the next index, unless a cancel since the last one dropped
    # pieces, in which case it is later.
    last_index = -1
    cancelled_since = False
    for event in events:
        if id(event) in owner:
            i = owner[id(event)]
            if event["type"] == "started":
                last_index = -1
                cancelled_since = False
            elif event["type"] == "cancelled" and messages[i][0] == "cancel":
                cancelled_since = True
            if event["type"] == "done":
                # Everything sent up to the `end` was spoken, and nothing sent after it.
                assert current == segment_of[i], (seed, "done in the wrong segment", i)
                assert spoken.get(current, "") == sent_up_to(messages, segment_of, i), (seed, i)
            else:
                current = segment_of[i]
        elif event["type"] == "speaking":
            if cancelled_since:
                assert event["_index"] > last_index, (seed, "piece index went backwards")
            else:
                assert event["_index"] == last_index + 1, (seed, "piece index skipped")
            last_index = event["_index"]
            cancelled_since = False
            spoken[current] = spoken.get(current, "") + letters(event["text"])
            speaking_in.setdefault(current, []).append(event)
            if event["_aborted"]:
                aborted_in.add(current)

    for k in set(segment_of):
        said, got = spoken.get(k, ""), sent.get(k, "")
        assert got.startswith(said), (seed, k, "spoken text is not a prefix of the sent text")
        ends_at = next((j for j in range(len(messages)) if segment_of[j] == k + 1), None)
        if supersedes(ends_at):
            continue
        # Not cancelled: no piece may be lost or cut short.
        assert k not in aborted_in, (seed, k, "piece aborted without a cancel")
        assert len(said) >= len(flushed.get(k, "")), (seed, k, "lost uncancelled text")


    # BC-36, from this checker's own model: when start i was applied, was a piece of the
    # segment it ends queued or in flight? In flight: spoken before i and not yet ended. Queued:
    # letters drained but not yet spoken. Everything up to the last flush/end was drained for
    # sure; later text maybe, so "pending" has a floor and a ceiling. A start that sent
    # `cancelled` needs the ceiling to allow pending work, one that didn't the floor to rule it
    # out.
    for i, (kind, _) in enumerate(messages):
        if kind != "start" or i == 0:
            continue
        k = segment_of[i] - 1
        before = [e for e in speaking_in.get(k, []) if e["_at"] <= i]
        in_flight = any(e["_ended_at"] > i for e in before)
        spoken_before = len("".join(letters(e["text"]) for e in before))
        surely_pending = in_flight or len(flushed.get(k, "")) > spoken_before
        maybe_pending = in_flight or len(sent.get(k, "")) > spoken_before
        if i in interrupting:
            assert maybe_pending, (seed, i, "start cancelled with nothing pending")
        else:
            assert not surely_pending, (seed, i, "start left pending work uncancelled")
    return len(interrupting)


def sent_up_to(messages: list[tuple[str, str]], segment_of: list[int], i: int) -> str:
    k = segment_of[i]
    return "".join(text for j, (_, text) in enumerate(messages[: i + 1]) if segment_of[j] == k)


def test_sc_005_random_sequences_keep_done_and_cancelled_exact() -> None:
    """SC-005: 1,000 seeded sequences of text/flush/cancel/end/start/instruction with random
    worker progress between them."""
    totals = {
        "cancelled": 0, "done": 0, "speaking": 0, "error": 0, "aborted": 0, "interrupting_start": 0
    }
    for seed in SEEDS:
        messages, events = run_sequence(seed)
        totals["interrupting_start"] += check_sequence(seed, messages, events)
        for e in events:
            if e["type"] in totals:
                totals[e["type"]] += 1
            if e.get("_aborted"):
                totals["aborted"] += 1
    # The sequences really exercise the interesting paths, not just the idle ones.
    assert all(count > 100 for count in totals.values()), totals
