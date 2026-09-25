"""`reference_audio` tests (specs/003-cpp-compatible-api/research.md R9).

Each malformed-input test's docstring names the C++/BC defect it stands in for: BC-11
(bad audio silently became voice design instead of a `400`), BC-15 (a hand-written WAV
parser that could over-read or divide by zero, and 8/24-bit PCM that decoded as
silence), BC-16 (unlimited-length or too-short clips silently accepted).

Valid inputs are generated in-test with `soundfile` (an already-present dependency,
used here for real rather than faked) so the round trip through libsndfile is real.
"""

from __future__ import annotations

import io

import librosa
import numpy as np
import pytest
import soundfile as sf

from breeze_infer.errors import ApiError
from breeze_infer.limits import MAX_REF_SECONDS
from breeze_infer.reference_audio import decode, predicted_frames
from tests.fakes import CODEC_SAMPLE_RATE, CODEC_SAMPLES_PER_FRAME, codec_frame_count

_SR = 16000


def _sine(seconds: float, sample_rate: int, channels: int = 1) -> np.ndarray:
    """A small, non-silent waveform -- silence would still decode, but a real signal
    makes it obvious the samples (not just the shape) survived the round trip."""
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    wav = np.tile(tone[:, None], (1, channels))
    return wav.astype(np.float64)


def _write(wav: np.ndarray, sample_rate: int, *, format: str, subtype: str | None = None) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=format, subtype=subtype)
    return buf.getvalue()


# --- valid inputs: every format/subtype the contract promises to accept -----------


@pytest.mark.parametrize(
    "subtype",
    ["PCM_U8", "PCM_16", "PCM_24", "FLOAT"],
)
def test_decodes_wav_pcm_and_float_subtypes_to_mono_float32(subtype: str) -> None:
    wav = _sine(0.5, _SR)
    blob = _write(wav, _SR, format="WAV", subtype=subtype)

    audio = decode(blob)

    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.sample_rate == _SR
    assert audio.samples.shape[0] == wav.shape[0]
    # Not silence: BC-15's 8/24-bit-decodes-as-silence defect, made concrete.
    assert np.abs(audio.samples).max() > 0.01


def test_decodes_stereo_44100hz_downmixed_to_mono() -> None:
    wav = _sine(0.5, 44100, channels=2)
    blob = _write(wav, 44100, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.ndim == 1
    assert audio.sample_rate == 44100
    assert audio.samples.shape[0] == wav.shape[0]
    # The downmix is the mean over channels; both channels are identical here, so the
    # mean must equal either one (loosely, given PCM_16 quantization).
    assert np.abs(audio.samples - wav[:, 0].astype(np.float32)).max() < 1e-3


def test_decodes_flac() -> None:
    wav = _sine(0.5, _SR)
    blob = _write(wav, _SR, format="FLAC", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.dtype == np.float32
    assert audio.sample_rate == _SR


def test_decodes_ogg() -> None:
    wav = _sine(0.5, _SR)
    blob = _write(wav, _SR, format="OGG", subtype="VORBIS")

    audio = decode(blob)

    assert audio.samples.dtype == np.float32
    assert audio.sample_rate == _SR


def test_duration_and_predicted_frames_are_populated() -> None:
    wav = _sine(1.0, _SR)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.duration_seconds == pytest.approx(1.0, abs=1e-6)
    assert audio.predicted_frames == predicted_frames(wav.shape[0], _SR)
    assert audio.predicted_frames >= 1


# --- invalid_audio: unreadable, empty, or an out-of-range header -------------------


def test_empty_blob_is_invalid_audio() -> None:
    """BC-11: an empty upload must be a 400, not silently-no-reference."""
    with pytest.raises(ApiError) as exc_info:
        decode(b"")
    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_audio"
    assert exc_info.value.message == "could not read ref_audio"


def test_garbage_bytes_are_invalid_audio() -> None:
    """BC-15: no hand-written parser to confuse -- libsndfile's own 'format not
    recognised' RuntimeError must map to invalid_audio, not crash or hang."""
    with pytest.raises(ApiError) as exc_info:
        decode(b"not audio, just noise" * 50)
    assert exc_info.value.code == "invalid_audio"


def test_over_max_bytes_is_invalid_audio() -> None:
    wav = _sine(0.1, _SR)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob, max_bytes=len(blob) - 1)
    assert exc_info.value.code == "invalid_audio"


def test_nine_channels_is_invalid_audio() -> None:
    """libsndfile itself accepts up to 256 channels; the 1-8 bound is ours."""
    wav = _sine(0.2, _SR, channels=9)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "invalid_audio"


def test_one_hertz_sample_rate_is_invalid_audio() -> None:
    """research.md R9: libsndfile accepts a sample rate of 1; our header check must
    reject it (it's below the 8-192 kHz range the contract promises)."""
    wav = _sine(10.0, 1)
    blob = _write(wav, 1, format="WAV", subtype="FLOAT")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "invalid_audio"


def test_mp3_is_invalid_audio() -> None:
    """MP3 is excluded from the contract's format list even though this libsndfile
    build can write and read it."""
    wav = _sine(0.2, _SR)
    blob = _write(wav, _SR, format="MP3")
    assert sf.info(io.BytesIO(blob)).format == "MP3"  # confirms this is a real MP3 fixture

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "invalid_audio"


# --- audio_too_long / audio_too_short ----------------------------------------------


def test_over_max_ref_seconds_is_audio_too_long() -> None:
    wav = _sine(MAX_REF_SECONDS + 1, _SR)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_long"
    assert exc_info.value.message == "ref_audio is longer than 30 seconds"


def test_zero_samples_is_audio_too_short() -> None:
    """BC-16: a clip too short for even one codec frame must be a 400, not a
    silently-ignored reference.

    `predicted_frames`'s formula is two nested ceiling divisions, so it is *never*
    less than 1 for any nonzero sample count -- one real sample at any sample rate
    still rounds up to a whole codec frame (`test_predicted_frames_matches_fake_codec`
    checks this over a wide grid). The only input that is actually "too short" is a
    file with zero decoded samples: a well-formed header around no sample data at
    all, distinct from the empty-blob and garbage-bytes cases above (both of which
    fail before any header is even parsed).
    """
    wav = np.zeros((0, 1), dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"


def test_one_sample_decodes_successfully_at_exactly_one_frame() -> None:
    """The flip side of the ceiling-math note above: a single real sample is valid
    input, not an error -- it just predicts the smallest possible reference (1 frame)."""
    wav = np.full((1, 1), 0.5, dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.shape[0] == 1
    assert audio.predicted_frames == 1


# --- predicted_frames must agree with the fake codec's own formula ----------------


def test_predicted_frames_matches_fake_codec() -> None:
    """`tests/fakes.py`'s `codec_frame_count` implements the same formula
    independently (it must not import production code); this asserts the two agree
    over a grid of rates and lengths, including ones that force resampling and ones
    that land exactly on a frame boundary."""
    sample_rates = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000)
    lengths = (0, 1, 2, 1920, 1919, 1921, 3840, 44100 * 30, 192000 * 30)
    for sr in sample_rates:
        for n in lengths:
            assert predicted_frames(n, sr) == codec_frame_count(n, sr), (n, sr)


def test_predicted_frames_matches_actual_librosa_resample() -> None:
    """Root-cause test (T021 review 2, finding #5): `test_predicted_frames_matches_fake_codec`
    above only checks `predicted_frames` against another implementation of the same
    formula -- two independent wrong implementations could still agree with each other.
    This instead compares against a real call to `librosa.resample`, mirroring qwen_tts's
    exact call (`librosa.resample(y=a, orig_sr=int(sr), target_sr=target_sr)`, no explicit
    `res_type`): librosa's `fix=True` default forces the returned length to exactly
    `ceil(len(y) * target_sr / orig_sr)` no matter which resampler backend runs, so this
    checks the framing step and the real resample step together."""
    sample_rates = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400)
    lengths = (1, 100, 1920, 1921, 3840, 3841, 44100 * 30)
    for sr in sample_rates:
        for n in lengths:
            wav = np.zeros(n, dtype=np.float32)
            if sr == CODEC_SAMPLE_RATE:
                resampled_len = n  # qwen_tts skips the call entirely at this rate too.
            else:
                resampled_len = len(librosa.resample(y=wav, orig_sr=sr, target_sr=CODEC_SAMPLE_RATE))
            expected = -(-resampled_len // CODEC_SAMPLES_PER_FRAME)
            assert predicted_frames(n, sr) == expected, (n, sr)
