"""One WebSocket connection's session: a pure state machine, no I/O, no GPU, no locks.

data-model.md "WebSocket Session" and research.md R15. `ws_server.py` (T077) owns the socket:
its reader task passes each parsed client message to `apply` and sends the immediate events it
returns; its one worker coroutine drains the work deque with `next_item` and reports each piece's
end with `mark_piece_done`. Both run on the same event loop, so no call here ever races another.

Why a work deque of markers rather than flags: every `cancel` puts exactly one `CancelMark` in
the deque, every `end` exactly one `EndMark`, every accepted `start` exactly one `StartMark`
(plus one `CancelMark` when it interrupts work). The worker handles them in order, only after
the piece before them is finished, so each client message gets exactly one reply, in message
order, after the audio it follows (BC-34, BC-35, BC-36). C++ latched a cancel flag instead,
which could go unacknowledged, fire twice, or swallow the next piece.

Stopping the piece in flight works through epochs: every `cancel` and every accepted `start`
begins a new epoch, and the worker checks `is_stale(piece)` between audio chunks.

A superseded `done` needs no epoch check: a `cancel` (or interrupting `start`) removes its own
session's EndMarks from the deque, and only those, so an EndMark still in the deque always owes
its `done`, including one a previous session left queued when a `start` interrupted nothing.
The worker sends it as soon as `next_item` returns it, with no `apply` in between.

Text is cut by the shared segmenter (`text_split.segment`, R11), so a text gives the same pieces
over HTTP and WebSocket.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from breeze_infer.http_fields import (
    DEFAULT_CFG_SCALE,
    DEFAULT_INSTRUCTION,
    DEFAULT_SEED,
)
from breeze_infer.limits import ANCHOR_CHARS, MAX_TEXT_CHARS
from breeze_infer.text_split import segment
from breeze_infer.ws_messages import (
    Cancel,
    End,
    Flush,
    Instruction,
    Message,
    Start,
    Text,
)

if TYPE_CHECKING:
    from breeze_infer.voice_registry import ResolvedVoice

_SEED_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class SessionConfig:
    """What `start` fixed for the session, with every default applied: all the worker needs
    to generate a piece. `instruction` is the one field that changes afterwards (the
    `instruction` message replaces the session's config), and the worker reads it from
    `Session.config` when each piece starts, so it applies to pieces that start later."""

    voice_id: str  # "" when the session has no voice; echoed in `started`
    voice: ResolvedVoice | None  # the reference, from `lookup_voice`
    ref_text_override: str | None  # `start`'s `ref_text`: replaces the voice's stored text
    instruction: str
    cfg_scale: float
    seed: int
    temperature: float | None  # None: the model default (FR-006)
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    max_new_tokens: int | None
    split_chars: int  # 0: no length splitting (BC-38)

    def piece_seed(self, index: int) -> int:
        """Piece `index`, counted from `start`, uses `(seed + index) mod 2^32` (ws-api.md)."""
        return (self.seed + index) & _SEED_MASK


@dataclass(frozen=True)
class Piece:
    """Text to speak. `index` counts from `start`, through cancels, so seeds do too."""

    epoch: int
    index: int
    text: str


@dataclass(frozen=True)
class EndMark:
    """Send `done`. A cancel that supersedes it removes it from the deque instead."""

    epoch: int


@dataclass(frozen=True)
class CancelMark:
    """Send `cancelled`. Never stale: every `cancel` gets its reply."""


@dataclass(frozen=True)
class StartMark:
    """Send `started` for this config."""

    config: SessionConfig


WorkItem = Piece | EndMark | CancelMark | StartMark


def _error(code: str, message: str, request_type: str) -> dict:
    return {"type": "error", "code": code, "message": message, "request_type": request_type}


def _instruction_or_default(instruction: str | None) -> str:
    """BC-37 (and BC-09 on HTTP): an absent or blank instruction means the default one."""
    if instruction is None or not instruction.strip():
        return DEFAULT_INSTRUCTION
    return instruction


def _request_type(message: Message) -> str:
    """The client's `type` for an error's `request_type`. Each ws_messages class is named for
    its wire type (`Start` for `"start"`, ...), so the name gives it without a second table
    to keep in step with the parser."""
    return type(message).__name__.lower()


class Session:
    """The session state of one connection (data-model.md "WebSocket Session").

    `lookup_voice` resolves a `start`'s `voice_id` (`VoiceRegistry.lookup`: the voice, or
    `None` when there is none). `default_split_chars` is the launch setting a `start` without
    `split_chars` uses: `ws_messages.parse` has no settings, so it leaves that field `None`.

    Public state the server reads: `config` (`None` until a valid `start`), `anchor` (the
    reference built from the first piece that succeeded, for the later pieces), and `work`
    (the deque itself; the worker takes items only through `next_item`).
    """

    def __init__(
        self,
        *,
        lookup_voice: Callable[[str], ResolvedVoice | None],
        default_split_chars: int,
    ) -> None:
        self._lookup_voice = lookup_voice
        self._default_split_chars = default_split_chars
        self.config: SessionConfig | None = None
        self.anchor: object | None = None
        self.work: deque[WorkItem] = deque()
        self._epoch = 0
        self._buffer = ""
        self._piece_index = 0
        # The piece `next_item` last handed out, until `mark_piece_done`: `start` needs it to
        # know whether work is pending, and `mark_piece_done` to know which piece ended.
        self._in_flight: Piece | None = None

    # ------------------------------------------------------------------ client messages

    def apply(self, message: Message) -> list[dict]:
        """Apply one parsed client message. Returns the events to send at once (`error`,
        `instruction_set`); `started`, `cancelled` and `done` go through the work deque
        instead, so they keep their place after the audio queued before them."""
        request_type = _request_type(message)
        if isinstance(message, Start):
            return self._start(message)
        if self.config is None:
            return [_error("not_started", "send start first", request_type)]
        if isinstance(message, Instruction):
            instruction = _instruction_or_default(message.instruction)
            self.config = replace(self.config, instruction=instruction)
            return [{"type": "instruction_set"}]
        if isinstance(message, Cancel):
            self._cancel()
            return []
        return self._add_text(message, request_type)

    def _start(self, message: Start) -> list[dict]:
        voice = None
        if message.voice_id is not None:
            voice = self._lookup_voice(message.voice_id)
            if voice is None:
                # Rejected before anything changes: the previous session carries on.
                return [_error("unknown_voice", "unknown voice_id", "start")]
        config = SessionConfig(
            voice_id=message.voice_id or "",
            voice=voice,
            ref_text_override=message.ref_text,
            instruction=_instruction_or_default(message.instruction),
            cfg_scale=DEFAULT_CFG_SCALE if message.cfg_scale is None else message.cfg_scale,
            seed=DEFAULT_SEED if message.seed is None else message.seed,
            temperature=message.temperature,
            top_k=message.top_k,
            top_p=message.top_p,
            repetition_penalty=message.repetition_penalty,
            max_new_tokens=message.max_new_tokens,
            split_chars=(
                self._default_split_chars if message.split_chars is None else message.split_chars
            ),
        )
        # BC-36: only a piece still to be spoken counts as interrupted work. A queued EndMark
        # alone keeps its `done` (sent before `started`); buffered text alone is dropped by the
        # reset below without a `cancelled`.
        if self.config is not None and self._piece_pending():
            self._cancel()
        # A new session, a new epoch: a later `cancel` then removes only this session's items,
        # never an EndMark the previous one left queued.
        self._epoch += 1
        self.config = config
        self._buffer = ""
        self.anchor = None
        self._piece_index = 0
        self.work.append(StartMark(config))
        return []

    def _cancel(self) -> None:
        """Drop this session's work not yet spoken, stop the piece in flight, reply once.

        Only items of the current epoch go: dropping this session's EndMarks is what makes a
        later `cancel` replace a pending `done`, but an EndMark a previous session left queued
        (older epoch) keeps its `done`. The anchor and the piece index survive: seeds keep
        counting from `start`, and a voice the session already anchored stays anchored."""
        self.work = deque(
            item
            for item in self.work
            if not (isinstance(item, Piece | EndMark) and item.epoch == self._epoch)
        )
        self._epoch += 1
        self._buffer = ""
        self.work.append(CancelMark())

    def _add_text(self, message: Text | Flush | End, request_type: str) -> list[dict]:
        text = message.text or ""
        if len(self._buffer) + len(text) > MAX_TEXT_CHARS:
            # BC-40, and a rejected message has no other effect: no drain, no EndMark.
            return [_error("text_too_long", "text is too long", request_type)]
        self._buffer += text
        # `text` speaks only what is complete (BC-39: a stop at the very end waits);
        # `flush` and `end` speak everything.
        self._drain(final=not isinstance(message, Text))
        if isinstance(message, End):
            self.work.append(EndMark(self._epoch))
        return []

    def _drain(self, *, final: bool) -> None:
        assert self.config is not None
        pieces, self._buffer = segment(
            self._buffer,
            budget=self.config.split_chars,
            first_budget=self._opening_budget(),
            final=final,
        )
        for text in pieces:
            self.work.append(Piece(self._epoch, self._piece_index, text))
            self._piece_index += 1

    # ------------------------------------------------------------------ opening budget

    def _opening_budget(self) -> int:
        """`segment`'s `first_budget`: a short opening piece while one is pending, so the
        first audio comes soon and the later pieces get an anchor (R11). Never more than
        `split_chars`, as on HTTP (`routes_speech._opening_budget`), and none without length
        splitting."""
        assert self.config is not None
        if not self.opening_pending or self.config.split_chars <= 0:
            return 0
        return min(ANCHOR_CHARS, self.config.split_chars)

    @property
    def opening_pending(self) -> bool:
        """data-model's `opening_pending`: no reference, no anchor yet, and no current piece
        queued or in flight that could still provide one.

        Derived rather than stored, so each of the data model's rules falls out of the state
        itself: the opening piece being queued turns it off; cancelling that piece before it
        anchored, or its failing with nothing else queued, turns it back on; an anchor, or a
        reference, keeps it off."""
        assert self.config is not None
        return self.config.voice is None and self.anchor is None and not self._piece_pending()

    def _piece_pending(self) -> bool:
        """A piece of the current epoch is queued or in flight."""
        in_flight = self._in_flight is not None and not self.is_stale(self._in_flight)
        return in_flight or any(isinstance(item, Piece) for item in self.work)

    # ------------------------------------------------------------------ worker side

    def next_item(self) -> WorkItem | None:
        """The next work item, or `None` when there is nothing to do. The worker handles one
        item at a time, and calls `mark_piece_done` for every `Piece` this returns before
        asking again."""
        if not self.work:
            return None
        item = self.work.popleft()
        if isinstance(item, Piece):
            self._in_flight = item
        return item

    def is_stale(self, item: Piece | EndMark) -> bool:
        """For a piece: true once a `cancel` or a `start` has begun a new epoch, so the piece
        stops at its next chunk (or is skipped).

        An EndMark is never stale: a cancel that supersedes one removes it from the deque, so
        any EndMark `next_item` returns owes its `done`. (Accepted because ws_server's worker
        asks about both.)"""
        if isinstance(item, EndMark):
            return False
        return item.epoch != self._epoch

    def mark_piece_done(self, anchor: object | None) -> None:
        """The piece in flight has ended, however it ended: finished, failed, cut short or
        skipped as stale. `anchor` is the reference built from it when it succeeded, else
        `None`.

        The first current piece to succeed provides the anchor; a stale piece's is ignored,
        since its session was cancelled or replaced."""
        piece, self._in_flight = self._in_flight, None
        if piece is None or self.is_stale(piece):
            return
        if self.anchor is None and anchor is not None:
            self.anchor = anchor
