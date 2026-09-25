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
   libsndfile raises for it can be the real end of the data -- but the same error
   also fires partway through a stream that libsndfile silently gave up decoding
   after real corruption, producing a plausible-looking but truncated result. Two
   checks make that unlikely (`_require_genuine_end`), skipped when the recovered
   length is already going to be rejected as too short (point 5) regardless: the
   underlying byte stream must be almost fully consumed, and -- for FLAC, which
   always falls back to an uncompressed subframe for data it can't compress -- the
   size of its audio frames (the file size *after* its metadata blocks -- a large
   comment or padding block must not count against this) can't be far more than an
   uncompressed encoding of the samples actually recovered would need, at a
   per-frame overhead bounded by STREAMINFO's own minimum block size. Each block is
   read into a buffer pre-filled with NaN, so even a block that ends in failure
   loses none of the real samples it did manage to decode before that: the first
   NaN marks exactly where decoding stopped. Known limit: a stream cut exactly at a
   frame boundary -- corruption or truncation that happens to leave what's left a
   complete, well-formed (if shorter) FLAC stream in its own right -- can't be told
   apart from a genuinely short recording without an independently known length,
   and isn't; the checks above catch the much more common case of a cut that lands
   *inside* a frame or past the metadata it needs to make sense.
4. A finite-value, then a magnitude check: a float WAV can carry NaN/Inf, or a value
   so large it would overflow when the codec processes it, and soundfile decodes
   either without complaint.
5. A minimum-length check: the reference clip must resample to at least one full 80 ms
   codec frame. `predicted_frames` alone can't express this -- its two nested ceiling
   divisions mean it never reports fewer than 1 frame for any nonzero input -- so the
   minimum is its own exact check, done in integer arithmetic on the native sample
   count rather than through the (rounding) resample step. This takes priority over
   point 3's checks: a clip that recovers under the minimum is always audio_too_short,
   never invalid_audio, whatever its plausibility.

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
# with the stream fully consumed, so this is a strict, mostly-defensive margin
# rather than one doing the bulk of the work (see `_require_genuine_end`).
_END_OF_STREAM_POSITION_MARGIN_BYTES = 64

# FLAC always has a verbatim (uncompressed) fallback for a subframe it can't
# compress, so a fully-decoded FLAC's *audio frames* (the file, minus its metadata
# blocks -- see `_flac_metadata_length_and_min_blocksize`) can never be far more
# than this many bytes per sample, per channel, at its own bit depth -- used as a
# sanity ceiling in `_require_genuine_end`. Unlisted subtypes fall back to 4
# (float/double), the largest real case.
_FLAC_VERBATIM_BYTES_PER_SAMPLE = {"PCM_S8": 1, "PCM_U8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4}

# A per-frame allowance for FLAC frame overhead, added on top of the verbatim
# sample bytes so the plausibility bound scales with how many frames the audio
# actually took, not a fixed guess (a stream encoded in many small frames --
# ffmpeg's -frame_size, say -- has proportionally more header/footer bytes to
# account for). From the frame format (https://xiph.org/flac/format.html#frame_header):
# a fixed 4-byte core, up to a 7-byte UTF-8-style frame/sample number, up to 2
# extra bytes each for an uncoded block size or sample rate, and an 8-bit CRC --
# 16 bytes in the worst case -- plus a 16-bit (2-byte) frame CRC footer.
_FLAC_FRAME_HEADER_MAX_BYTES = 16
_FLAC_FRAME_FOOTER_CRC_BYTES = 2
# Each channel's own subframe header (type + wasted-bits flag) adds up to about
# this many bytes; generously rounded rather than derived bit-for-bit.
_FLAC_SUBFRAME_HEADER_MAX_BYTES_PER_CHANNEL = 2


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


def _is_too_short(num_samples: int, sample_rate: int) -> bool:
    """The exact 80 ms minimum (module docstring, point 5): equivalent to
    `num_samples / sample_rate < 1920 / 24000`, cross-multiplied into integer
    arithmetic so there's no rounding at the boundary. Shared between `decode`'s own
    rejection and `_read_untrustworthy`'s early exit, which must agree with it
    exactly -- a clip this short is always audio_too_short, never invalid_audio,
    whatever `_require_genuine_end` would have made of it (point 5 takes priority
    over point 3)."""
    return num_samples * _CODEC_SAMPLE_RATE < _CODEC_SAMPLES_PER_FRAME * sample_rate


def _flac_metadata_length_and_min_blocksize(blob: bytes) -> tuple[int, int]:
    """Where a FLAC's metadata ends and its audio frames begin, plus STREAMINFO's
    minimum block size in samples -- so `_require_genuine_end`'s plausibility bound
    applies only to the audio frames (a large comment or padding block must not
    count against it) and scales its per-frame overhead allowance with the real
    frame count instead of a fixed guess.

    Metadata block layout (https://xiph.org/flac/format.html#format_overview): the
    4-byte `fLaC` magic is followed by one or more blocks, each a 4-byte header (a
    1-bit last-block flag, a 7-bit block type, a 24-bit big-endian length) then
    that many bytes of block data. STREAMINFO (type 0) is always first; its own
    first 2 bytes are the 16-bit minimum block size.

    Falls back to `(0, 0)` -- no metadata skipped, no known block size -- for
    anything that doesn't look like a well-formed FLAC. That's a defensive fallback
    only: `sf.SoundFile` has already confirmed the format by the time this runs.
    """
    if len(blob) < 4 + 4 + 2 or blob[:4] != b"fLaC":
        return 0, 0

    min_blocksize = int.from_bytes(blob[8:10], "big")
    offset = 4
    while offset + 4 <= len(blob):
        block_header = blob[offset]
        is_last = bool(block_header & 0x80)
        block_length = int.from_bytes(blob[offset + 1 : offset + 4], "big")
        offset += 4 + block_length
        if is_last:
            break
    return min(offset, len(blob)), min_blocksize


def _require_genuine_end(f: sf.SoundFile, raw: io.BytesIO, blob: bytes, filled: int, channels: int) -> None:
    """After a tolerated read failure (module docstring, point 3), rule out the read
    having merely given up on real corruption rather than reached a genuine
    unknown-length stream's end -- both raise the identical error. Two checks, each
    measured against real ffmpeg-piped FLACs (clean and corrupted):

    - the underlying byte stream must be almost fully consumed -- a read that failed
      on genuinely bad data partway through can still leave a meaningful amount of
      the blob unread, where a real end always leaves (at most) a tiny remainder;
    - for FLAC, the size of the audio frames (the file, minus its metadata blocks)
      can't be far more than an uncompressed encoding of `filled` samples would
      need, at a per-frame overhead bounded by STREAMINFO's minimum block size --
      since FLAC always falls back to a verbatim subframe for data it can't
      compress, a file whose audio frames are much bigger than that, for the amount
      of audio actually recovered, means real data was lost.

    Raises `ApiError(400, "invalid_audio", ...)` if either check fails.
    """
    gap = len(blob) - raw.tell()
    if gap > _END_OF_STREAM_POSITION_MARGIN_BYTES:
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    if f.format != "FLAC":
        return

    metadata_bytes, min_blocksize = _flac_metadata_length_and_min_blocksize(blob)
    audio_bytes = max(0, len(blob) - metadata_bytes)

    # `min_blocksize` bounds the true frame count from above (every real frame is
    # at least this many samples), so this is a safe -- if anything, generous --
    # estimate of how much per-frame overhead to allow.
    frame_count = -(-filled // min_blocksize) if min_blocksize > 0 else filled
    per_frame_overhead_bytes = frame_count * (
        _FLAC_FRAME_HEADER_MAX_BYTES + _FLAC_FRAME_FOOTER_CRC_BYTES + channels * _FLAC_SUBFRAME_HEADER_MAX_BYTES_PER_CHANNEL
    )
    bytes_per_sample = _FLAC_VERBATIM_BYTES_PER_SAMPLE.get(f.subtype, 4)
    max_plausible_audio_bytes = filled * channels * bytes_per_sample + per_frame_overhead_bytes

    if audio_bytes > max_plausible_audio_bytes:
        raise ApiError(400, "invalid_audio", "could not read ref_audio")


def _downmix_into(mono: np.ndarray, filled: int, chunk: np.ndarray, actual: int) -> None:
    """`mono[filled:filled+actual] = mean(chunk[:actual], axis=1)`, in float64: two
    channels near float32's ~3.4e38 max would otherwise overflow a float32 sum
    before the mean is even taken."""
    mono[filled : filled + actual] = chunk[:actual].mean(axis=1, dtype=np.float64)


def _read_trustworthy(f: sf.SoundFile, mono: np.ndarray, max_samples: int) -> int:
    """Read a file whose declared length is trustworthy (module docstring, point 3):
    a plain blockwise read straight into `mono` (owned and preallocated by the
    caller), no NaN-sentinel bookkeeping -- any read failure here is a hard error
    (`decode`'s `except RuntimeError`), never tolerated. Returns the number of
    samples filled.
    """
    filled = 0
    while filled < max_samples:
        request = min(_READ_BLOCK_FRAMES, max_samples - filled)
        block = f.read(frames=request, dtype="float32", always_2d=True)
        actual = block.shape[0]
        if actual == 0:
            break
        _downmix_into(mono, filled, block, actual)
        filled += actual
        if actual < request:
            break  # a normal short read: the true end of a trustworthy-length file
    return filled


def _read_untrustworthy(
    f: sf.SoundFile, raw: io.BytesIO, blob: bytes, mono: np.ndarray, max_samples: int, channels: int
) -> int:
    """Read a file whose declared length can't be trusted (module docstring, point
    3): the same blockwise read into `mono` (owned and preallocated by the caller),
    but with the NaN-sentineled recovery buffer -- allocated once here, since only
    this path ever needs it -- and the tolerant handling of the stream's real end.
    Returns the number of samples filled.
    """
    sample_rate = f.samplerate
    chunk = np.empty((_READ_BLOCK_FRAMES, channels), dtype=np.float32)
    filled = 0
    while filled < max_samples:
        request = min(_READ_BLOCK_FRAMES, max_samples - filled)
        # NaN-filled rather than left as-is: a read that fails partway still writes
        # every sample it did decode into this buffer before raising, and NaN can't
        # be a real decoded value, so the first NaN row marks exactly where
        # decoding stopped (`read`'s `out=` returns the array itself on success,
        # but nothing at all when it raises).
        view = chunk[:request]
        view.fill(np.nan)
        try:
            result = f.read(frames=request, dtype="float32", always_2d=True, out=view)
            actual = result.shape[0]
        except RuntimeError as exc:
            if getattr(exc, "code", None) != _SEEK_PAST_REAL_END_CODE:
                raise
            nan_rows = np.isnan(view).any(axis=1)
            actual = int(np.argmax(nan_rows)) if nan_rows.any() else view.shape[0]
            _downmix_into(mono, filled, view, actual)
            filled += actual
            if not _is_too_short(filled, sample_rate):
                _require_genuine_end(f, raw, blob, filled, channels)
            break
        if actual == 0:
            break
        _downmix_into(mono, filled, view, actual)
        filled += actual
        if actual < request:
            break
    return filled


def _open_and_read(blob: bytes) -> tuple[np.ndarray, int]:
    """Validate the header and decode, bounded so neither an untrustworthy frame
    count nor an enormous real file can allocate more than `MAX_REF_SECONDS` worth of
    memory. Raises `ApiError` directly for a bad header, an already-known
    over-length file, or a tolerated read failure that turns out not to be a genuine
    end (`_require_genuine_end`); any other failure is left as a `RuntimeError` for
    the caller to translate, since both the open and any `read()` call can fail that
    way.
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

        mono = np.empty(max_samples, dtype=np.float32)
        if trustworthy_length:
            filled = _read_trustworthy(f, mono, max_samples)
        else:
            filled = _read_untrustworthy(f, raw, blob, mono, max_samples, f.channels)

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

    if _is_too_short(actual_num_samples, sample_rate):
        raise ApiError(400, "audio_too_short", "ref_audio is too short")

    return DecodedAudio(
        samples=mono,
        sample_rate=sample_rate,
        duration_seconds=actual_num_samples / sample_rate,
        predicted_frames=predicted_frames(actual_num_samples, sample_rate),
    )
