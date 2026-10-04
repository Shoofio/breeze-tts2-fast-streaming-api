"""Launch configuration: a frozen ``Settings`` value object built from argv.

Per data-model.md "Settings", this is built once at the composition root from
the command line. It never reads ``os.environ`` (Principle III: environment
resolution, such as ``$TORCHINDUCTOR_CACHE_DIR``, belongs to the composition
root that consumes ``compile_cache_dir``, not to argument parsing).
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from breeze_infer.origins import canonical_origin

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SPLIT_CHARS = 600
DEFAULT_CHUNK_FIRST = 1
DEFAULT_CHUNK_MAX = 25
DEFAULT_VOICES_DIR = Path("voices")
DEFAULT_ATTN_IMPLEMENTATION = "eager"

# The smallest Apple Silicon Mac with room for the bf16 weights, the codec and the KV cache
# (research R5).
MLX_MIN_MEMORY_BYTES = 16 * 1024**3

# Where to get each backend's weights; named in the checkpoint/backend mismatch refusals.
# The MLX command names the 8-bit repo because 8-bit is the Mac default (spec FR-012).
MLX_DOWNLOAD_COMMAND = (
    "uvx --from huggingface_hub hf download mlx-community/Breeze-TTS-2-mlx-8bit "
    "--revision c6e4a2ff6ab9afba68b7853de802273ffe23fb49"
)
PYTORCH_DOWNLOAD_COMMAND = "uvx --from huggingface_hub hf download BreezeBlue/Breeze-TTS-2"

_PYTORCH_MODEL_TYPE = "breeze"
_MLX_MODEL_TYPE = "breeze_tts"

# Sentinel distinguishing "--ws-port" absent (derive port + 1) from an
# explicit "--ws-port disabled" (which resolves to None on Settings).
_WS_PORT_UNSET = object()


@dataclass(frozen=True)
class Platform:
    """The machine facts the backend choice depends on, read once at the entry point.

    Passed into ``settings_from_args`` rather than read there, so the parser stays a pure
    function of its arguments and tests build a ``Platform`` directly instead of patching.
    """

    system: str
    machine: str
    memory_bytes: int

    @classmethod
    def detect(cls) -> Platform:
        # Memory only matters for the MLX backend, which only runs on macOS. os.sysconf doesn't
        # exist on Windows, and every launch calls this, so elsewhere memory is left at 0.
        memory_bytes = 0
        if sys.platform == "darwin":
            memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return cls(system=sys.platform, machine=platform_module.machine(), memory_bytes=memory_bytes)

    @property
    def is_macos(self) -> bool:
        return self.system == "darwin"

    @property
    def is_apple_silicon(self) -> bool:
        return self.is_macos and self.machine == "arm64"


@dataclass(frozen=True)
class CheckpointKind:
    """What a checkpoint directory holds, from its ``config.json``."""

    format: Literal["pytorch", "mlx"]
    weights: Literal["bf16", "8bit"]


def checkpoint_kind(config: dict[str, Any], directory: Path | str = "checkpoint") -> CheckpointKind:
    """Classify a parsed ``config.json``; ``ValueError`` carries the refusal message.

    ``directory`` is only there so the message can name the checkpoint the user passed.
    """
    model_type = config.get("model_type")
    if model_type == _PYTORCH_MODEL_TYPE:
        return CheckpointKind("pytorch", "bf16")
    if model_type != _MLX_MODEL_TYPE:
        raise ValueError(
            f"{directory} has model_type {model_type!r}; expected {_PYTORCH_MODEL_TYPE!r} "
            f"(PyTorch) or {_MLX_MODEL_TYPE!r} (MLX)"
        )
    quantization = config.get("quantization")
    if quantization is None:
        return CheckpointKind("mlx", "bf16")
    # A hand-edited config could hold anything here; refuse it like any other unsupported
    # quantization rather than crash the launch with a traceback.
    if not isinstance(quantization, dict):
        quantization = {}
    if quantization.get("bits") == 8 and quantization.get("mode") == "mxfp8":
        return CheckpointKind("mlx", "8bit")
    raise ValueError(
        f"{directory} is {quantization.get('bits')}-bit {quantization.get('mode')}; "
        "the MLX backend supports bf16 and 8-bit (mxfp8)"
    )


@dataclass(frozen=True)
class Settings:
    """Launch configuration. Every field is already validated.

    Every field is required: ``Settings`` is only ever built by
    ``settings_from_args``, which fills in every default and runs every
    check, so there is no partially-defaulted instance to accidentally ship.
    Tests that need a ``Settings`` build one through ``settings_from_args``
    too, not by constructing this dataclass directly.

    ``ws_port`` is ``None`` when the WebSocket server is disabled (either by
    ``--ws-port disabled`` or, going forward, any other way disabling is
    spelled); otherwise it is a validated port distinct from ``port``.

    ``cors`` is empty when CORS is off (the default); ``("*",)`` when any
    origin is allowed; otherwise a tuple of validated, deduplicated origins
    (see ``_parse_cors_origins``).
    """

    model_path: Path
    host: str
    port: int
    ws_port: int | None
    cors: tuple[str, ...]
    split_chars: int
    chunk_first: int
    chunk_max: int
    voices_dir: Path
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool
    attn_implementation: str
    compile_cache_dir: Path | None
    backend: Literal["cuda", "mlx"]
    weights: Literal["bf16", "8bit"]


# --port and --ws-port must be plain unsigned integers: no leading '+' or
# '-', no decimal point, no exponent, no surrounding whitespace, and no
# non-ASCII digit (e.g. full-width '８０８０' or Arabic-Indic '٨٠٨٠') -- int()
# itself accepts all of those, so the strict, ASCII-only shape is enforced
# separately. An explicit [0-9] class (rather than \d with re.ASCII) keeps
# that ASCII-only intent visible at the call site.
_STRICT_PORT_RE = re.compile(r"[0-9]+")


class _CudaOnlyBoolean(argparse.BooleanOptionalAction):
    """A ``--x/--no-x`` flag that records it was given, for the MLX backend's refusal.

    Recording the option string (rather than comparing against the default) keeps argv order
    for the message and also catches ``--no-x``, which leaves the default value unchanged.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        _note_cuda_only(namespace, option_string)
        super().__call__(parser, namespace, values, option_string)


class _CudaOnlyStore(argparse.Action):
    """A store action that records it was given; see ``_CudaOnlyBoolean``."""

    def __call__(self, parser, namespace, values, option_string=None):
        _note_cuda_only(namespace, option_string)
        setattr(namespace, self.dest, values)


def _note_cuda_only(namespace: argparse.Namespace, option_string: str | None) -> None:
    given = getattr(namespace, "cuda_only_given", ())
    namespace.cuda_only_given = (*given, option_string)


def _parse_port(value: str) -> int:
    """argparse ``type=`` for ``--port``: a plain, strictly-digit port number."""
    if not _STRICT_PORT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"--port must be a plain number (got {value!r})"
        )
    return int(value)


def _parse_ws_port(value: str) -> int | None:
    """argparse ``type=`` for ``--ws-port``: a plain digit port number, or ``disabled``."""
    if value.strip().lower() == "disabled":
        return None
    if not _STRICT_PORT_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"--ws-port must be a port number or 'disabled' (got {value!r})"
        )
    return int(value)


def _parse_cors_origins(value: str | None) -> tuple[str, ...]:
    """``None`` (flag absent) disables CORS; ``"*"`` (bare flag) allows any origin.

    Otherwise ``value`` is a comma-separated list. Each entry is trimmed,
    validated and canonicalized as a bare origin (``canonical_origin``, in
    ``origins.py`` so ``cors.py``'s ``CorsMiddleware`` can canonicalize an
    incoming ``Origin`` header the same way before comparing) and deduplicated, preserving
    first-seen order -- so ``http://a.example:80`` and ``http://a.example``
    collapse to one entry. ``"*"`` means any origin only when every entry is
    ``"*"`` (so ``"*,*"`` collapses to ``("*",)``); mixed with any other
    origin it is rejected, as is an allowlist that is empty after trimming.
    """
    if value is None:
        return ()
    raw_entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not raw_entries:
        # An empty allowlist would silently mean "CORS off" despite the flag. No "--cors" here:
        # `settings_from_args` prepends the flag name once, uniformly, for every rejection this
        # function raises (review-agent final pass, issue 3).
        raise ValueError("empty origin list")
    if "*" in raw_entries:
        if any(entry != "*" for entry in raw_entries):
            raise ValueError("'*' can't be combined with other origins")
        return ("*",)
    origins: list[str] = []
    seen: set[str] = set()
    for entry in raw_entries:
        normalized = canonical_origin(entry)
        if normalized not in seen:
            seen.add(normalized)
            origins.append(normalized)
    return tuple(origins)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model_path", type=Path, help="Path to the model directory")
    parser.add_argument(
        "--backend",
        choices=("cuda", "mlx"),
        default=None,
        help="Inference backend (default: mlx on an Apple Silicon Mac, else cuda)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address for HTTP and WebSocket (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=DEFAULT_PORT,
        help=f"HTTP port (default: {DEFAULT_PORT})",
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
            "comma-separated allowlist (default: disabled). Takes an "
            "optional value, so pass it after model_path, or use "
            "--cors=ORIGINS, to avoid swallowing the next argument"
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
        "--fast-all", action=_CudaOnlyBoolean, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=_CudaOnlyBoolean, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=_CudaOnlyBoolean, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=_CudaOnlyBoolean, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=_CudaOnlyBoolean, default=False
    )
    parser.add_argument(
        "--fast-codec", action=_CudaOnlyBoolean, default=False
    )
    parser.add_argument(
        "--attn-implementation",
        action=_CudaOnlyStore,
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
        action=_CudaOnlyStore,
        type=Path,
        default=None,
        help=(
            "Where torch.compile artifacts persist between starts "
            "(default: $TORCHINDUCTOR_CACHE_DIR if set, else ./.cache/torchinductor)"
        ),
    )
    return parser


def _read_config(model_path: Path) -> dict[str, Any] | None:
    """The checkpoint's ``config.json``, or ``None`` when it is missing or not a JSON object."""
    try:
        config = json.loads((model_path / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _choose_backend(
    parser: argparse.ArgumentParser, args: argparse.Namespace, platform: Platform | None
) -> Literal["cuda", "mlx"]:
    """The backend to run, refusing what this machine can't serve (FR-002, FR-003, FR-004).

    ``platform`` is ``None`` for callers that never read the machine (every existing test),
    which means "not a Mac": the default is cuda, as before this option existed.
    """
    # Any Mac defaults to mlx, so an Intel Mac is told it needs Apple Silicon rather than
    # being sent to a CUDA backend that macOS can't run either.
    on_macos = platform is not None and platform.is_macos
    backend = args.backend or ("mlx" if on_macos else "cuda")

    if backend == "cuda":
        if on_macos:
            parser.error("--backend cuda is not available on macOS; use --backend mlx (the default here)")
        return "cuda"

    if platform is None or not platform.is_apple_silicon:
        machine = "unknown" if platform is None else f"{platform.system} {platform.machine}"
        parser.error(f"--backend mlx needs an Apple Silicon Mac (this machine: {machine})")
    if platform.memory_bytes < MLX_MIN_MEMORY_BYTES:
        gigabytes = round(platform.memory_bytes / 1024**3, 1)
        parser.error(f"--backend mlx needs at least 16 GB of memory (this machine: {gigabytes:g} GB)")
    cuda_only = tuple(dict.fromkeys(getattr(args, "cuda_only_given", ())))
    if cuda_only:
        parser.error(f"{', '.join(cuda_only)} only apply to --backend cuda; remove them")
    return "mlx"


def _check_checkpoint(
    parser: argparse.ArgumentParser,
    model_path: Path,
    backend: Literal["cuda", "mlx"],
) -> Literal["bf16", "8bit"]:
    """Refuse a checkpoint that doesn't match the backend; return its weight precision.

    The CUDA path stays as lenient as it was before this check existed: a missing or odd
    ``config.json`` is not its concern, and only a checkpoint that says it is MLX is refused.
    The MLX backend reads the precision from the file, so there it must exist.
    """
    config = _read_config(model_path)
    if backend == "cuda":
        if config is not None and config.get("model_type") == _MLX_MODEL_TYPE:
            parser.error(
                f"{model_path} holds MLX weights; --backend cuda needs: {PYTORCH_DOWNLOAD_COMMAND}"
            )
        return "bf16"

    if config is None:
        parser.error(
            f"{model_path} has no readable config.json; --backend mlx needs the MLX weights: "
            f"{MLX_DOWNLOAD_COMMAND}"
        )
    try:
        kind = checkpoint_kind(config, model_path)
    except ValueError as exc:
        parser.error(str(exc))
    if kind.format == "pytorch":
        parser.error(
            f"{model_path} is the PyTorch checkpoint; --backend mlx needs the MLX weights: "
            f"{MLX_DOWNLOAD_COMMAND}"
        )
    return kind.weights


def settings_from_args(
    argv: Sequence[str] | None = None, platform: Platform | None = None
) -> Settings:
    """Parse ``argv`` (``sys.argv[1:]`` when ``None``) into a validated ``Settings``.

    Every rejection goes through ``parser.error()``, which prints the usage
    and message to stderr and raises ``SystemExit(2)`` -- the same failure
    mode as an argparse-level type error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.model_path.is_dir():
        parser.error(f"model_path must be an existing directory (got {args.model_path})")

    backend = _choose_backend(parser, args, platform)
    weights = _check_checkpoint(parser, args.model_path, backend)

    port = args.port
    if not 1 <= port <= 65535:
        parser.error(f"--port must be between 1 and 65535 (got {port})")

    ws_port = args.ws_port
    ws_port_derived = ws_port is _WS_PORT_UNSET
    if ws_port_derived:
        ws_port = port + 1
    if ws_port is not None:
        if not 1 <= ws_port <= 65535:
            if ws_port_derived:
                parser.error(
                    f"the derived --ws-port (--port + 1 = {ws_port}) is out of "
                    "range; pass --ws-port to set it explicitly"
                )
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
        # `_parse_cors_origins`/`canonical_origin` (origins.py) keep their own messages neutral
        # about which flag they're validating -- `canonical_origin` is also run on every request's
        # incoming `Origin` header, where "--cors" would be meaningless -- so this is the one place
        # that actually knows the value came from `--cors` (review-agent final pass, issue 3).
        parser.error(f"--cors: {exc}")

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
        backend=backend,
        weights=weights,
    )
