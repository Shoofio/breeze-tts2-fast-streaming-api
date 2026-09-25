"""Reference-audio decoding (specs/003-cpp-compatible-api/research.md R9).

An uploaded `ref_audio` part is untrusted bytes: possibly empty, possibly garbage,
possibly a well-formed header describing something the codec can't use. `decode()`
turns it into a bounds-checked mono waveform in the order research.md R9 requires:

1. A size cap, before anything is parsed.
2. A header-only check via `sf.info` -- format, channel count, sample rate, duration --
   so a malformed file is rejected without libsndfile ever decoding sample data (the
   hand-written WAV parser this replaces is BC-15's defect).
3. Only then the real decode (`sf.read`) and a downmix to mono.
4. A predicted-frame check, since a clip too short to become at least one codec frame
   is silently no reference at all (BC-16) -- rather than a distinct "empty" case,
   this is where a well-formed but zero-sample file (or a corrupted one libsndfile
   clamped down to zero samples) actually gets rejected.

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


@dataclass(frozen=True)
class DecodedAudio:
    """A validated reference clip, ready for the codec's own encode step."""

    samples: np.ndarray  # mono float32, shape (num_samples,)
    sample_rate: int
    duration_seconds: float
    predicted_frames: int


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
       this must not be simplified to a single division.
    2. Frame it: a ceiling division by `encode_downsample_rate` (1920 samples/frame).

    tests/fakes.py's `codec_frame_count` implements this same formula independently for
    `FakeCodec`; `test_predicted_frames_matches_fake_codec` asserts the two agree over a
    grid of rates and lengths, and T049 asserts this formula against the real encode on
    the GPU.
    """
    if duration_samples <= 0:
        return 0
    sample_rate = int(sample_rate)
    if sample_rate != _CODEC_SAMPLE_RATE:
        ratio = float(_CODEC_SAMPLE_RATE) / sample_rate
        duration_samples = int(np.ceil(duration_samples * ratio))
    return int(np.ceil(duration_samples / _CODEC_SAMPLES_PER_FRAME))


def decode(blob: bytes, *, max_bytes: int = MAX_AUDIO_BYTES) -> DecodedAudio:
    """Decode and validate an uploaded `ref_audio` part.

    Raises `ApiError(400, ...)` for everything the contract rejects: unreadable or
    empty input, an oversized upload, an out-of-range header field, a clip over
    `MAX_REF_SECONDS`, or one too short to produce a single codec frame. `max_bytes`
    is a parameter (rather than reading `limits.MAX_AUDIO_BYTES` directly) purely so
    tests can exercise the size cap without a 25 MiB fixture.
    """
    if not blob or len(blob) > max_bytes:
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    try:
        info = sf.info(io.BytesIO(blob))
    except RuntimeError:  # LibsndfileError's base class (research.md R9).
        raise ApiError(400, "invalid_audio", "could not read ref_audio") from None

    # A malformed header -- a truncated `data` chunk, say -- doesn't raise here:
    # libsndfile clamps a bogus length to what's actually present (research.md R9),
    # so `info.frames` comes back as a small or zero count instead. That's not
    # distinguished from a well-formed but genuinely empty/near-empty file (both just
    # report a low frame count), so both fall through to the same place: the
    # predicted-frame check below, once real sample data has been read.
    if (
        info.format not in _ALLOWED_FORMATS
        or not (_MIN_CHANNELS <= info.channels <= _MAX_CHANNELS)
        or not (_MIN_SAMPLE_RATE <= info.samplerate <= _MAX_SAMPLE_RATE)
    ):
        raise ApiError(400, "invalid_audio", "could not read ref_audio")

    duration_seconds = info.frames / info.samplerate
    if duration_seconds > MAX_REF_SECONDS:
        raise ApiError(400, "audio_too_long", "ref_audio is longer than 30 seconds")

    samples, sample_rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)

    frames = predicted_frames(mono.shape[0], sample_rate)
    if frames < 1:
        raise ApiError(400, "audio_too_short", "ref_audio is too short")

    return DecodedAudio(
        samples=mono,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        predicted_frames=frames,
    )
