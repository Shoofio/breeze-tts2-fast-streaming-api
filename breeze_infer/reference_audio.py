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
   libsndfile raises for it is treated as the real end of the data, once the
   underlying byte stream is almost fully consumed -- skipped when the recovered
   length is already going to be rejected as too short (point 5) regardless. Each
   block is read into a buffer pre-filled with NaN, so even a block that ends in
   failure loses none of the real samples it did manage to decode before that: the
   first NaN row marks exactly where decoding stopped. **Known limit**: for a FLAC
   with unknown length, a stream cut short -- between frames or inside one -- can be
   accepted with fewer samples than the original; only genuinely corrupted data is
   rejected (a decoder-level failure, distinct from the tolerated one above, which
   libsndfile raises reliably -- a FLAC frame parser of our own was considered and
   rejected as not worth the complexity for the residual risk). A trustworthy-length
   file has no such gap: any read failure there is always a hard error.
4. A finite-value, then a magnitude check: a float WAV can carry NaN/Inf, or a value
   so large it would overflow when the codec processes it, and soundfile decodes
   either without complaint.
5. A minimum-length check: the reference clip must resample to at least one full 80 ms
   codec frame. `predicted_frames` alone can't express this -- its two nested ceiling
   divisions mean it never reports fewer than 1 frame for any nonzero input -- so the
   minimum is its own exact check, done in integer arithmetic on the native sample
   count rather than through the (rounding) resample step. This takes priority over
   point 3's end-of-stream check: a clip that recovers under the minimum is always
   audio_too_short, whatever the byte stream looks like.

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

# How many frames to pull from soundfile per read() call, for a trustworthy-length
# file and for the bulk of an untrustworthy-length one alike: bounded so a single
# call never needs to hold more than this regardless of the file's real or claimed
# length -- only the running mono buffer scales with duration. A 23 MiB, 29.9 s,
# 192 kHz stereo piped FLAC (incompressible noise, the worst case for this path)
# decodes in well under a second at this block size.
_READ_BLOCK_FRAMES = 65_536

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

# How much of the underlying byte stream is allowed to remain unread when a
# tolerated read failure fires, still counted as "the real end". Measured against
# real ffmpeg-piped FLACs (clean and corrupted): the failure consistently arrives
# with the stream fully consumed, so this is a strict, mostly-defensive margin.
# A size-based plausibility bound (an upper limit on compressed bytes per sample,
# derived from FLAC's verbatim-subframe fallback) was tried here too, but review
# found it both too strict (it rejected valid ffmpeg output -- a short final frame
# near the end of a `-frame_size`-encoded stream) and too loose (a cut inside a
# frame can still look plausible) -- removed rather than chasing it with a FLAC
# frame parser of our own; see the module docstring's "Known limit".
_END_OF_STREAM_POSITION_MARGIN_BYTES = 64


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


# The most codec frames an accepted reference can become: `decode` accepts at most
# MAX_REF_SECONDS * sample_rate samples, which resample to MAX_REF_SECONDS * 24 kHz --
# plus one sample, because `_resampled_length` ceils a product computed with a float
# ratio, and at some rates (191,995 Hz, for one) that product lands a hair above the
# whole number. 30 s is 375 frames, and that extra sample makes it 376. voice_file uses
# this to bound a saved voice's `frames` to what POST /v1/voices could ever have
# produced.
MAX_REF_FRAMES = predicted_frames(MAX_REF_SECONDS * _CODEC_SAMPLE_RATE + 1, _CODEC_SAMPLE_RATE)


def _is_too_short(num_samples: int, sample_rate: int) -> bool:
    """The exact 80 ms minimum (module docstring, point 5): equivalent to
    `num_samples / sample_rate < 1920 / 24000`, cross-multiplied into integer
    arithmetic so there's no rounding at the boundary. Shared between `decode`'s own
    rejection and `_read_samples`'s early exit, which must agree with it exactly --
    a clip this short is always audio_too_short, whatever the byte stream looks
    like (point 5 takes priority over point 3)."""
    return num_samples * _CODEC_SAMPLE_RATE < _CODEC_SAMPLES_PER_FRAME * sample_rate


def _read_samples(
    f: sf.SoundFile, raw: io.BytesIO, blob: bytes, max_samples: int, channels: int, tolerate_end: bool
) -> np.ndarray:
    """Read a file's audio in bounded blocks straight into a preallocated mono
    buffer, downmixing each block immediately rather than materializing the full
    multi-channel array (module docstring, point 3). Returns `mono[:filled]`
    (a view, not a copy -- `_open_and_read` owns turning it into one).

    `tolerate_end` is True only for a file whose declared length can't be trusted.
    Every block is read into a buffer pre-filled with NaN and passed as `out=`, on
    both paths alike: a read that fails partway still writes every sample it did
    decode into that buffer before raising, and NaN can't be a real decoded value,
    so the first NaN row marks exactly where decoding stopped (`read`'s `out=`
    returns the array itself on success, but nothing at all when it raises). When
    `tolerate_end` is True, a specific internal error libsndfile raises for
    reading past a stream's true end -- matched by its numeric code, since the
    message isn't a stable contract -- is treated as that real end once the
    underlying byte stream is almost fully consumed, *unless* the recovered length
    is already going to be rejected as too short (point 5 takes priority). It's
    never tolerated when False, and no other failure (e.g. a genuine decode error)
    is ever tolerated either way.
    """
    mono = np.empty(max_samples, dtype=np.float32)
    chunk = np.empty((_READ_BLOCK_FRAMES, channels), dtype=np.float32)
    filled = 0
    while filled < max_samples:
        request = min(_READ_BLOCK_FRAMES, max_samples - filled)
        view = chunk[:request]
        view.fill(np.nan)
        try:
            result = f.read(frames=request, dtype="float32", always_2d=True, out=view)
            actual = result.shape[0]
        except RuntimeError as exc:
            if not tolerate_end or getattr(exc, "code", None) != _SEEK_PAST_REAL_END_CODE:
                raise
            nan_rows = np.isnan(view).any(axis=1)
            actual = int(np.argmax(nan_rows)) if nan_rows.any() else view.shape[0]
            # float64 accumulation: two channels near float32's ~3.4e38 max would
            # otherwise overflow a float32 sum before the mean is even taken.
            mono[filled : filled + actual] = view[:actual].mean(axis=1, dtype=np.float64)
            filled += actual
            if not _is_too_short(filled, f.samplerate):
                gap = len(blob) - raw.tell()
                if gap > _END_OF_STREAM_POSITION_MARGIN_BYTES:
                    raise ApiError(400, "invalid_audio", "could not read ref_audio") from None
            break
        if actual == 0:
            break
        mono[filled : filled + actual] = view[:actual].mean(axis=1, dtype=np.float64)
        filled += actual
        if actual < request:
            break  # a normal short read: the true end of a trustworthy-length file
    return mono[:filled]


def _open_and_read(blob: bytes) -> tuple[np.ndarray, int]:
    """Validate the header and decode, bounded so neither an untrustworthy frame
    count nor an enormous real file can allocate more than `MAX_REF_SECONDS` worth
    of memory. Raises `ApiError` directly for a bad header, an already-known
    over-length file, or a tolerated read failure that turns out not to have
    reached the end after all (`_read_samples`); any other failure is left as a
    `RuntimeError` for the caller to translate, since both the open and any
    `read()` call can fail that way.
    """
    raw = io.BytesIO(blob)
    with sf.SoundFile(raw) as f:
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

        mono = _read_samples(f, raw, blob, max_samples, f.channels, tolerate_end=not trustworthy_length)

    return mono.copy(), sample_rate  # a copy, not a view into the 30 s buffer


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

    if _is_too_short(actual_num_samples, sample_rate):
        raise ApiError(400, "audio_too_short", "ref_audio is too short")

    return DecodedAudio(
        samples=mono,
        sample_rate=sample_rate,
        duration_seconds=actual_num_samples / sample_rate,
        predicted_frames=predicted_frames(actual_num_samples, sample_rate),
    )
