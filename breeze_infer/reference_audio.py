"""Reference-audio decoding (specs/003-cpp-compatible-api/research.md R9).

An uploaded `ref_audio` part is untrusted bytes: possibly empty, possibly garbage,
possibly a well-formed header describing something the codec can't use, possibly a
header whose declared length can't be trusted at all. `decode()` turns it into a
bounds-checked mono waveform:

1. A size cap, before anything is parsed.
2. One `sf.SoundFile` open, covering both the header check (format, channel count,
   sample rate) and the decode. A single `except RuntimeError` around the whole
   open-and-read handles every way libsndfile can fail on a bad file: an OGG
   truncated mid-stream fails to open at all, while a FLAC truncated mid-stream opens
   fine (its metadata block is intact) and only fails once the frame data runs out.
3. A bounded, blockwise decode straight into a preallocated mono buffer, downmixing
   each block immediately rather than materializing the full multi-channel array, so
   peak memory is `MAX_REF_SECONDS * sample_rate` regardless of channel count or
   upload size. `sf.info`'s `frames` field is used to reject an over-length file
   without decoding it, but only when it looks like a real value -- a FLAC written by
   a streaming encoder with no known length upfront (STREAMINFO `total_samples` left
   at 0, e.g. one piped from `ffmpeg`) reports `frames` as `INT64_MAX` instead of
   raising. For such a file, the actual decoded length decides instead, found by
   reading up to one sample past the cap and stopping there -- enough to tell "too
   long" from "ends right at the limit" without trusting the header or reading
   anything beyond it. Ending early is normal for this kind of file (the encoder
   didn't know how long the stream would run), so a specific internal error
   libsndfile raises for it is treated as the real end of the data once at least one
   block has already been read; a decoder-level corruption error is not.
4. A finite-value, then a magnitude check: a float WAV can carry NaN/Inf, or a value
   so large it would overflow when the codec processes it, and soundfile decodes
   either without complaint.
5. A minimum-length check: the reference clip must resample to at least one full 80 ms
   codec frame. `predicted_frames` alone can't express this -- its two nested ceiling
   divisions mean it never reports fewer than 1 frame for any nonzero input -- so the
   minimum is its own exact check, done in integer arithmetic on the native sample
   count rather than through the (rounding) resample step.

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

# A sample this large is finite but nonsensical for real audio (PCM/float references
# stay within +/-1.0 by convention) and can overflow arithmetic downstream in the
# codec; rejected outright rather than clipped, since a client sending it is more
# likely confused or malicious than intending something recoverable.
_MAX_ABS_SAMPLE = 8.0

# How many frames to pull from soundfile per read() call in the common case, where
# either the header's length is trustworthy or the file has at least this much data
# left. Bounded so a block never needs to be bigger than this regardless of the
# file's real or claimed length -- only the running mono buffer scales with duration.
_READ_BLOCK_FRAMES = 65_536

# The block size used once a file's length can't be trusted (see module docstring,
# point 3). Deliberately much smaller than `_READ_BLOCK_FRAMES`: reading past the
# real end of such a stream is tolerated, but only after at least one successful
# read, and only up to a whole block's worth of the tail can be lost when that
# happens -- a small block keeps that loss to a fraction of a second even for the
# shortest accepted clip, at the cost of more (still cheap) read() calls.
_UNKNOWN_LENGTH_READ_FRAMES = 256

# `sf.SoundFile.frames` above this can't be a real file's length at any sample rate
# this module accepts -- it's libsndfile's way of reporting a length it doesn't
# actually know (observed as INT64_MAX, i.e. 2**63 - 1, for a FLAC whose STREAMINFO
# total_samples was left at 0), or some other header corruption. Below this, `frames`
# is trusted for sizing and an early rejection when it's already over the cap; at or
# above it, the actual decoded length decides everything instead.
_MAX_PLAUSIBLE_FRAMES = 2**40

# libsndfile's own numeric error code for "Internal psf_fseek() failed." -- confirmed
# empirically as the failure a read raises when it runs past the true end of a stream
# whose declared length couldn't be trusted in the first place. Matched by code
# rather than message text, since the message isn't a stable contract.
_SEEK_PAST_REAL_END_CODE = 39


def _max_samples(sample_rate: int) -> int:
    """One sample past `MAX_REF_SECONDS` at `sample_rate` -- both the read cap the
    decode loop stops at and the threshold the final length is compared against, kept
    as one formula so the two can't drift apart."""
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
    """Validate the header and decode, bounded so neither an untrustworthy frame
    count nor an enormous real file can allocate more than `MAX_REF_SECONDS` worth of
    memory. Raises `ApiError` directly for a bad header or an already-known
    over-length file; any other failure is left as a `RuntimeError` for the caller to
    translate, since both the open and any `read()` call can fail that way.
    """
    with sf.SoundFile(io.BytesIO(blob)) as f:
        if (
            f.format not in _ALLOWED_FORMATS
            or not (_MIN_CHANNELS <= f.channels <= _MAX_CHANNELS)
            or not (_MIN_SAMPLE_RATE <= f.samplerate <= _MAX_SAMPLE_RATE)
        ):
            raise ApiError(400, "invalid_audio", "could not read ref_audio")

        sample_rate = f.samplerate
        max_samples = _max_samples(sample_rate)
        trustworthy_length = 0 <= f.frames < _MAX_PLAUSIBLE_FRAMES

        if trustworthy_length and f.frames >= max_samples:
            raise ApiError(400, "audio_too_long", "ref_audio is longer than 30 seconds")

        read_block_frames = _READ_BLOCK_FRAMES if trustworthy_length else _UNKNOWN_LENGTH_READ_FRAMES
        mono = np.empty(max_samples, dtype=np.float32)
        filled = 0
        while filled < max_samples:
            request = min(read_block_frames, max_samples - filled)
            try:
                block = f.read(frames=request, dtype="float32", always_2d=True)
            except RuntimeError as exc:
                already_have_data = filled > 0
                is_real_end_of_unknown_length_stream = (
                    not trustworthy_length and getattr(exc, "code", None) == _SEEK_PAST_REAL_END_CODE
                )
                if already_have_data and is_real_end_of_unknown_length_stream:
                    break
                raise
            if block.shape[0] == 0:
                break
            end = filled + block.shape[0]
            # float64 accumulation: two channels near float32's ~3.4e38 max would
            # otherwise overflow a float32 sum before the mean is even taken.
            mono[filled:end] = block.mean(axis=1, dtype=np.float64)
            filled = end

    return mono[:filled].copy(), sample_rate  # a copy, not a view into the 30 s buffer


def decode(blob: bytes, *, max_bytes: int = MAX_AUDIO_BYTES) -> DecodedAudio:
    """Decode and validate an uploaded `ref_audio` part.

    Raises `ApiError(400, ...)` for everything the contract rejects: unreadable or
    empty input, an oversized upload, an out-of-range header field, non-finite or
    absurdly large sample data, a clip over `MAX_REF_SECONDS`, or one under the
    minimum codec frame. `max_bytes` is a parameter (rather than reading
    `limits.MAX_AUDIO_BYTES` directly) purely so tests can exercise the size cap
    without a 25 MiB fixture.
    """
    if not blob or len(blob) > max_bytes:
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    try:
        mono, sample_rate = _open_and_read(blob)
    except RuntimeError:
        # sf.LibsndfileError -- soundfile's own open/read failures -- subclasses
        # RuntimeError, so this alone covers a bad-format open failure and a mid-read
        # decode failure (see _open_and_read's docstring for when each can happen).
        raise ApiError(400, "invalid_audio", "could not read ref_audio") from None

    if not np.isfinite(mono).all():
        raise ApiError(400, "invalid_audio", "could not read ref_audio")
    if np.any(np.abs(mono) > _MAX_ABS_SAMPLE):
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    actual_num_samples = mono.shape[0]
    if actual_num_samples >= _max_samples(sample_rate):
        raise ApiError(400, "audio_too_long", "ref_audio is longer than 30 seconds")

    # The exact 80 ms minimum, in integer arithmetic on the native sample count
    # (cross-multiplied rather than resampling first, so there's no rounding at the
    # boundary): equivalent to `actual_num_samples / sample_rate < 1920 / 24000`.
    if actual_num_samples * _CODEC_SAMPLE_RATE < _CODEC_SAMPLES_PER_FRAME * sample_rate:
        raise ApiError(400, "audio_too_short", "ref_audio is too short")

    return DecodedAudio(
        samples=mono,
        sample_rate=sample_rate,
        duration_seconds=actual_num_samples / sample_rate,
        predicted_frames=predicted_frames(actual_num_samples, sample_rate),
    )
