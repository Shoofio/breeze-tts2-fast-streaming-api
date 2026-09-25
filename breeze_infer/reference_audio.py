"""Reference-audio decoding (specs/003-cpp-compatible-api/research.md R9).

An uploaded `ref_audio` part is untrusted bytes: possibly empty, possibly garbage,
possibly a well-formed header describing something the codec can't use, possibly a
header that outright lies about its own length. `decode()` turns it into a
bounds-checked mono waveform in the order research.md R9 requires:

1. A size cap, before anything is parsed.
2. One `sf.SoundFile` open, covering both the header check (format, channel count,
   sample rate) and the decode -- a single `except (RuntimeError, sf.LibsndfileError)`
   around both, since a malformed file can fail at either point: an OGG truncated
   mid-stream fails to open at all ("file is malformed"), while a FLAC truncated
   mid-stream opens fine (its metadata block is intact) and only fails once the frame
   data runs out ("flac decoder lost sync"). Neither is distinguished from the other;
   both mean the upload can't be trusted (review-agent finding #1).
3. A bounded, blockwise decode straight to a preallocated mono buffer, downmixing
   each block immediately rather than materializing the full multi-channel array --
   peak memory is `MAX_REF_SECONDS * sample_rate` regardless of channel count or
   upload size (finding #4). The bound doubles as the length check: `sf.info`'s
   `frames` is never used for sizing, because a FLAC written by a streaming encoder
   (STREAMINFO `total_samples` left at 0, meaning "unknown") reports `frames` as
   `INT64_MAX` rather than raising (finding #3) -- reading one sample past the cap
   and stopping there tells "too long" from "ends right at the limit" without ever
   trusting that field, or reading anything the header might claim beyond it.
4. A finite-value check: a float WAV can carry NaN/Inf samples verbatim, and
   soundfile decodes them without complaint (finding #2).
5. A minimum-length check, separate from the decode: BC-16 (spec decision) sets the
   minimum reference clip at one full 80 ms codec frame -- 1,920 samples at the 24 kHz
   -equivalent (resampled) length, the same arithmetic `predicted_frames` uses for its
   own resample step. This does not follow from `predicted_frames` alone: that
   function's two nested ceiling divisions mean it never reports fewer than 1 frame
   for any nonzero input, so a "too short" rejection needs its own explicit floor.

FR-009 and contracts/http-api.md's `ref_audio` row and error messages.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from breeze_infer.errors import ApiError
from breeze_infer.limits import MAX_AUDIO_BYTES, MAX_REF_SECONDS

# contracts/http-api.md's ref_audio row: "WAV, WAVEX, FLAC or OGG; 1-8 channels;
# 8-192 kHz". These are ours, not libsndfile's -- it happily opens a 1 Hz mono file
# (research.md R9), so the header check has to reject that itself.
_ALLOWED_FORMATS = frozenset({"WAV", "WAVEX", "FLAC", "OGG"})
_MIN_CHANNELS, _MAX_CHANNELS = 1, 8
_MIN_SAMPLE_RATE, _MAX_SAMPLE_RATE = 8_000, 192_000

# The 12 Hz codec's own constants (qwen_tts.core.tokenizer_12hz's
# configuration_qwen3_tts_tokenizer_v2: input/output sample rate and
# encode_downsample_rate). tests/fakes.py's CODEC_SAMPLE_RATE / CODEC_SAMPLES_PER_FRAME
# are the same values, kept in sync by test_predicted_frames_matches_fake_codec rather
# than a shared import -- the fake must not depend on production code.
_CODEC_SAMPLE_RATE = 24_000
_CODEC_SAMPLES_PER_FRAME = 1920

# BC-16 (USER DECISION): the minimum reference clip is one full codec frame at the
# resampled length -- not "at least 1 predicted frame" (predicted_frames rounds up,
# so that's true of almost any nonzero clip). Same unit as _CODEC_SAMPLES_PER_FRAME
# by definition; kept as its own name so the two concerns (the minimum vs. the frame
# size used to count frames) read as separate even though they're numerically equal.
_MIN_RESAMPLED_SAMPLES = _CODEC_SAMPLES_PER_FRAME

# How many frames to pull from soundfile per read() call. Bounded so a block never
# needs to be bigger than this regardless of the file's real or claimed length --
# only the running mono buffer scales with duration, not any per-read allocation.
_READ_BLOCK_FRAMES = 65_536


def _max_samples(sample_rate: int) -> int:
    """One sample past `MAX_REF_SECONDS` at `sample_rate` -- both the read cap
    `_open_and_read` stops at and the threshold `decode` compares the actual decoded
    length against, kept as one formula so the two can't drift apart."""
    return MAX_REF_SECONDS * sample_rate + 1


@dataclass(frozen=True)
class DecodedAudio:
    """A validated reference clip, ready for the codec's own encode step."""

    samples: np.ndarray  # mono float32, shape (num_samples,)
    sample_rate: int
    duration_seconds: float
    predicted_frames: int


def _resampled_length(duration_samples: int, sample_rate: int) -> int:
    """The 12 Hz codec's own resample step, in isolation: `librosa.resample`'s
    ceil-ratio arithmetic (`predicted_frames`'s docstring has the full rationale).
    """
    if duration_samples <= 0:
        return 0
    sample_rate = int(sample_rate)
    if sample_rate == _CODEC_SAMPLE_RATE:
        return duration_samples
    ratio = float(_CODEC_SAMPLE_RATE) / sample_rate
    return int(np.ceil(duration_samples * ratio))


def predicted_frames(duration_samples: int, sample_rate: int) -> int:
    """How many 12 Hz codec frames `duration_samples` samples at `sample_rate` Hz becomes.

    Exactly the real tokenizer's arithmetic, in two steps (qwen_tts's
    `Qwen3TTSTokenizer._normalize_audio_inputs`/`load_audio`):

    1. Resample to 24 kHz, skipped entirely when `sample_rate` already is 24 kHz
       (`int(sr) != target_sr` guards the call). `librosa.resample` computes the
       resampled length as `int(np.ceil(n * ratio))` with `ratio = float(24000) / sr`
       computed *first*, as one float division, then multiplied by the sample count.
       That is not the same value as ceiling `n * 24000 / sr` computed the other order
       -- float rounding differs at some rates (44.1/22.05/11.025/88.2/176.4 kHz) -- so
       this must not be simplified to a single division (`_resampled_length` above).
    2. Frame it: a ceiling division by `encode_downsample_rate` (1920 samples/frame),
       computed as the integer identity `-(-n // d)` rather than
       `int(np.ceil(n / d))` -- the float version loses precision for `n` beyond
       2**53, silently returning the wrong frame count for a long enough clip.

    tests/fakes.py's `codec_frame_count` implements this same formula independently for
    `FakeCodec`; `test_predicted_frames_matches_fake_codec` asserts the two agree over a
    grid of rates and lengths, and T049 asserts this formula against the real encode on
    the GPU.
    """
    resampled_length = _resampled_length(duration_samples, sample_rate)
    return -(-resampled_length // _CODEC_SAMPLES_PER_FRAME)


def _open_and_read(blob: bytes) -> tuple[np.ndarray, int]:
    """Validate the header and decode, bounded so neither a lying frame count nor an
    enormous real file can allocate more than `MAX_REF_SECONDS` worth of memory.

    Raises `ApiError(400, "invalid_audio", ...)` for a bad header, and re-raises
    `RuntimeError`/`sf.LibsndfileError` for the caller to translate -- both the open
    and every `read()` call can fail that way (module docstring, point 2).
    """
    with sf.SoundFile(io.BytesIO(blob)) as f:
        if (
            f.format not in _ALLOWED_FORMATS
            or not (_MIN_CHANNELS <= f.channels <= _MAX_CHANNELS)
            or not (_MIN_SAMPLE_RATE <= f.samplerate <= _MAX_SAMPLE_RATE)
        ):
            raise ApiError(400, "invalid_audio", "could not read ref_audio")

        sample_rate = f.samplerate
        # `f.frames` isn't trusted for sizing (module docstring, point 3): the cap is
        # fixed at one sample past the limit, so reading up to it -- and no further --
        # is enough to tell "too long" from "ends right at the limit" either way.
        max_samples = _max_samples(sample_rate)
        mono = np.empty(max_samples, dtype=np.float32)
        filled = 0
        while filled < max_samples:
            block = f.read(
                frames=min(_READ_BLOCK_FRAMES, max_samples - filled),
                dtype="float32",
                always_2d=True,
            )
            if block.shape[0] == 0:
                break
            end = filled + block.shape[0]
            mono[filled:end] = block.mean(axis=1)
            filled = end

    return mono[:filled], sample_rate


def decode(blob: bytes, *, max_bytes: int = MAX_AUDIO_BYTES) -> DecodedAudio:
    """Decode and validate an uploaded `ref_audio` part.

    Raises `ApiError(400, ...)` for everything the contract rejects: unreadable or
    empty input, an oversized upload, an out-of-range header field, non-finite sample
    data, a clip over `MAX_REF_SECONDS`, or one under the minimum codec frame.
    `max_bytes` is a parameter (rather than reading `limits.MAX_AUDIO_BYTES` directly)
    purely so tests can exercise the size cap without a 25 MiB fixture.
    """
    if not blob or len(blob) > max_bytes:
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    try:
        mono, sample_rate = _open_and_read(blob)
    except (RuntimeError, sf.LibsndfileError):
        # sf.LibsndfileError already subclasses RuntimeError; named explicitly anyway
        # so the relationship is visible here rather than only in soundfile's MRO.
        # Covers an open failure (bad format, "file is malformed") and a read failure
        # mid-stream (FLAC's "lost sync" on a truncated file, or an internal seek
        # failure walking past the real data when a header overstates the length --
        # both confirmed empirically, not guessed; see the module docstring).
        raise ApiError(400, "invalid_audio", "could not read ref_audio") from None

    if not np.isfinite(mono).all():
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    actual_num_samples = mono.shape[0]
    if actual_num_samples >= _max_samples(sample_rate):
        raise ApiError(400, "audio_too_long", "ref_audio is longer than 30 seconds")

    if _resampled_length(actual_num_samples, sample_rate) < _MIN_RESAMPLED_SAMPLES:
        raise ApiError(400, "audio_too_short", "ref_audio is too short")

    return DecodedAudio(
        samples=mono,
        sample_rate=sample_rate,
        duration_seconds=actual_num_samples / sample_rate,  # from the decode, not a header field
        predicted_frames=predicted_frames(actual_num_samples, sample_rate),
    )
