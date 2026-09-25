"""Launch configuration: a frozen ``Settings`` value object built from argv.

Per data-model.md "Settings", this is built once at the composition root from
the command line. It never reads ``os.environ`` (Principle III: environment
resolution, such as ``$TORCHINDUCTOR_CACHE_DIR``, belongs to the composition
root that consumes ``compile_cache_dir``, not to argument parsing).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SPLIT_CHARS = 600
DEFAULT_CHUNK_FIRST = 1
DEFAULT_CHUNK_MAX = 25
DEFAULT_VOICES_DIR = Path("voices")
DEFAULT_ATTN_IMPLEMENTATION = "eager"

# Sentinel distinguishing "--ws-port" absent (derive port + 1) from an
# explicit "--ws-port disabled" (which resolves to None on Settings).
_WS_PORT_UNSET = object()


@dataclass(frozen=True)
class Settings:
    """Launch configuration. Every field is already validated.

    ``ws_port`` is ``None`` when the WebSocket server is disabled (either by
    ``--ws-port disabled`` or, going forward, any other way disabling is
    spelled); otherwise it is a validated port distinct from ``port``.

    ``cors`` is empty when CORS is off (the default); ``("*",)`` when any
    origin is allowed; otherwise a tuple of trimmed, exact-match origins.
    """

    model_path: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    ws_port: int | None = DEFAULT_PORT + 1
    cors: tuple[str, ...] = ()
    split_chars: int = DEFAULT_SPLIT_CHARS
    chunk_first: int = DEFAULT_CHUNK_FIRST
    chunk_max: int = DEFAULT_CHUNK_MAX
    voices_dir: Path = DEFAULT_VOICES_DIR
    fast_all: bool | None = None
    fast_text_encoder: bool = False
    fast_backbone_prefill: bool = False
    fast_backbone_decode: bool = False
    fast_depth_decoder: bool = False
    fast_codec: bool = False
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    compile_cache_dir: Path | None = None


def _parse_ws_port(value: str) -> int | None:
    """argparse ``type=`` for ``--ws-port``: a port number, or ``disabled``."""
    if value.strip().lower() == "disabled":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--ws-port must be a port number or 'disabled' (got {value!r})"
        ) from exc


def _parse_cors_origins(value: str | None) -> tuple[str, ...]:
    """``None`` (flag absent) disables CORS; ``"*"`` (bare flag) allows any origin.

    Ported from ``breeze_infer/api.py`` (``api-alignment`` branch). ``"*"``
    means any origin only when it is the whole value (``"*,"`` and ``" * "``
    count). Mixed with other origins it is rejected at startup.
    """
    if value is None:
        return ()
    if value == "*":
        return ("*",)
    origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())
    if origins == ("*",):
        return origins
    if "*" in origins:
        raise ValueError("'*' can't be combined with other origins")
    if not origins:
        # An empty allowlist would silently mean "CORS off" despite the flag.
        raise ValueError("--cors was given an empty origin list")
    return origins


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model_path", type=Path, help="Path to the model directory")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address for HTTP and WebSocket (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--ws-port",
        type=_parse_ws_port,
        default=_WS_PORT_UNSET,
        metavar="PORT|disabled",
        help=(
            "WebSocket port (default: HTTP port + 1); pass 'disabled' to "
            "turn off the WebSocket server"
        ),
    )
    parser.add_argument(
        "--cors",
        nargs="?",
        const="*",
        default=None,
        metavar="ORIGINS",
        help=(
            "Enable CORS: bare flag allows any origin, or pass a "
            "comma-separated allowlist (default: disabled)"
        ),
    )
    parser.add_argument(
        "--split-chars",
        type=int,
        default=DEFAULT_SPLIT_CHARS,
        metavar="N",
        help=(
            "Split long text at sentence-ish boundaries near this many "
            f"characters; 0 disables splitting (default: {DEFAULT_SPLIT_CHARS})"
        ),
    )
    parser.add_argument(
        "--chunk-first",
        type=int,
        default=DEFAULT_CHUNK_FIRST,
        metavar="N",
        help=f"Codec frames in the first streamed flush (default: {DEFAULT_CHUNK_FIRST})",
    )
    parser.add_argument(
        "--chunk-max",
        type=int,
        default=DEFAULT_CHUNK_MAX,
        metavar="N",
        help=f"Largest flush the stream ramps up to, in codec frames (default: {DEFAULT_CHUNK_MAX})",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=DEFAULT_VOICES_DIR,
        help=f"Directory holding saved voices (default: ./{DEFAULT_VOICES_DIR})",
    )
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--attn-implementation",
        # flash_attention_2 is deliberately absent: Hugging Face's FA2 path
        # rejects the backbone's 4D attention masks ("cu_seqlens_k must have
        # shape (batch_size + 1)") in both eager and CUDA-graph modes.
        choices=("eager", "sdpa"),
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help=(
            "Attention kernel for the backbone and text encoder; the fast "
            "text-encoder stage always runs sdpa for graph capture regardless "
            f"of this setting (default: {DEFAULT_ATTN_IMPLEMENTATION})"
        ),
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=None,
        help=(
            "Where torch.compile artifacts persist between starts "
            "(default: $TORCHINDUCTOR_CACHE_DIR if set, else ./.cache/torchinductor)"
        ),
    )
    return parser


def settings_from_args(argv: Sequence[str] | None = None) -> Settings:
    """Parse ``argv`` (``sys.argv[1:]`` when ``None``) into a validated ``Settings``.

    Every rejection goes through ``parser.error()``, which prints the usage
    and message to stderr and raises ``SystemExit(2)`` -- the same failure
    mode as an argparse-level type error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    port = args.port
    if not 1 <= port <= 65535:
        parser.error(f"--port must be between 1 and 65535 (got {port})")

    ws_port = args.ws_port
    if ws_port is _WS_PORT_UNSET:
        ws_port = port + 1
    if ws_port is not None:
        if not 1 <= ws_port <= 65535:
            parser.error(
                f"--ws-port must be between 1 and 65535, or 'disabled' (got {ws_port})"
            )
        if ws_port == port:
            parser.error(f"--ws-port must differ from --port (got {ws_port})")

    if args.split_chars < 0:
        parser.error(f"--split-chars must be >= 0 (got {args.split_chars})")

    if args.chunk_first < 1:
        parser.error(f"--chunk-first must be at least 1 (got {args.chunk_first})")
    if args.chunk_max < 1:
        parser.error(f"--chunk-max must be at least 1 (got {args.chunk_max})")
    # main.cpp parity: the first flush never streams past the ramp ceiling.
    chunk_first = min(args.chunk_first, args.chunk_max)

    try:
        cors = _parse_cors_origins(args.cors)
    except ValueError as exc:
        parser.error(str(exc))

    return Settings(
        model_path=args.model_path,
        host=args.host,
        port=port,
        ws_port=ws_port,
        cors=cors,
        split_chars=args.split_chars,
        chunk_first=chunk_first,
        chunk_max=args.chunk_max,
        voices_dir=args.voices_dir,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        attn_implementation=args.attn_implementation,
        compile_cache_dir=args.compile_cache_dir,
    )
