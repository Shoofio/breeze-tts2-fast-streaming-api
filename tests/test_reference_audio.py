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


def _flac_with_bogus_total_samples(wav: np.ndarray, sample_rate: int) -> bytes:
    """A real FLAC whose STREAMINFO ``total_samples`` field is zeroed out, which is
    what a streaming FLAC writer emits when the length isn't known upfront. libsndfile
    then reports ``frames`` as ``INT64_MAX`` (9223372036854775807) rather than raising
    -- confirmed empirically, this is review-agent's finding #3.

    Byte layout (https://xiph.org/flac/format.html#metadata_block_streaminfo): the
    ``fLaC`` magic (4 bytes) and one metadata block header (4 bytes) precede a fixed
    34-byte STREAMINFO block; the 36-bit ``total_samples`` field is the low nibble of
    STREAMINFO byte 13 plus all of STREAMINFO bytes 14-17.
    """
    data = bytearray(_write(wav, sample_rate, format="FLAC", subtype="PCM_16"))
    assert data[:4] == b"fLaC"
    field_start = 8 + 13
    data[field_start] &= 0xF0
    data[field_start + 1 : field_start + 5] = b"\x00\x00\x00\x00"
    return bytes(data)


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
    """7: opposite-signed channels prove the downmix actually averages -- identical
    channels (the old fixture) would still pass a downmix that silently just took one
    channel and ignored the other."""
    num_samples = int(0.5 * 44100)
    wav = np.empty((num_samples, 2), dtype=np.float64)
    wav[:, 0] = 0.2
    wav[:, 1] = -0.2
    blob = _write(wav, 44100, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.ndim == 1
    assert audio.sample_rate == 44100
    assert audio.samples.shape[0] == num_samples
    # Mean of +0.2 and -0.2 is 0; PCM_16 quantizes each channel independently, so the
    # tolerance is one quantization step (1/32768), not exact -- still two orders of
    # magnitude tighter than "silently took one channel" (which would fail at ~0.2).
    assert np.abs(audio.samples).max() < 1e-4


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

    # 5: duration comes from what was actually decoded, not a header field.
    assert audio.duration_seconds == pytest.approx(audio.samples.shape[0] / _SR)
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


def test_truncated_flac_is_invalid_audio() -> None:
    """1 HIGH: a FLAC whose STREAMINFO metadata parses fine but whose audio-frame data
    is cut off mid-stream fails during the *read*, not the open -- libsndfile's own
    wording is "flac decoder lost sync". Confirms the single try/except covers both
    the open and the read, not just the header-only stage."""
    wav = _sine(1.0, _SR)
    blob = _write(wav, _SR, format="FLAC", subtype="PCM_16")
    truncated = blob[: len(blob) // 2]

    with pytest.raises(ApiError) as exc_info:
        decode(truncated)
    assert exc_info.value.code == "invalid_audio"


def test_truncated_ogg_is_invalid_audio() -> None:
    """1 HIGH: an OGG truncated mid-stream fails to even open ("Supported file format
    but file is malformed") -- the other of the two failure points the combined
    except clause has to cover, alongside the FLAC read-time failure above."""
    wav = _sine(1.0, _SR)
    blob = _write(wav, _SR, format="OGG", subtype="VORBIS")
    truncated = blob[: len(blob) // 2]

    with pytest.raises(ApiError) as exc_info:
        decode(truncated)
    assert exc_info.value.code == "invalid_audio"


def test_flac_streaminfo_claiming_more_than_its_data_is_invalid_audio() -> None:
    """1 HIGH: a short, real FLAC whose STREAMINFO total_samples is corrupted to claim
    far more data than actually follows it. libsndfile can't cleanly signal EOF for
    this case -- walking past the real end of stream raises "Internal psf_fseek()
    failed" instead of returning a short read -- so it must map to invalid_audio via
    the same except clause as lost-sync above, not be treated as a stream we can
    happily read to EOF (that's the genuinely-long-file case below)."""
    wav = _sine(1.0, _SR)  # far below any cap; the point is the header, not the length
    blob = _flac_with_bogus_total_samples(wav, _SR)

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "invalid_audio"


def test_flac_unknown_length_header_enforces_actual_decoded_length() -> None:
    """3: STREAMINFO total_samples=0 makes libsndfile report frames as INT64_MAX -- an
    implausible value that must never be trusted for sizing. When the *real*
    underlying audio genuinely exceeds MAX_REF_SECONDS, the bounded read (capped at
    30 s + 1 sample, regardless of what the header claims) still catches it, without
    reading -- or allocating memory for -- anything like the INT64_MAX the header
    reports."""
    wav = _sine(MAX_REF_SECONDS + 1, _SR)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_long"
    assert exc_info.value.message == "ref_audio is longer than 30 seconds"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_samples_is_invalid_audio(bad_value: float) -> None:
    """2 HIGH: a float WAV can carry NaN/Inf verbatim -- soundfile doesn't reject it on
    write or read -- so decode() must reject it, rather than handing the codec
    non-finite input."""
    wav = _sine(0.1, _SR)
    wav[len(wav) // 2, 0] = bad_value
    blob = _write(wav, _SR, format="WAV", subtype="FLOAT")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_audio"
    assert exc_info.value.message == "could not read ref_audio"


def test_reads_across_internal_block_boundaries_correctly() -> None:
    """4: the decode loop reads in bounded blocks rather than one large sf.read call,
    so a clip spanning several block boundaries -- at the max channel count, where a
    downmix bug would be easiest to hide -- must still downmix to the exact right
    values everywhere, not just within the first block."""
    num_samples = 200_000  # several times any reasonable internal block size
    wav = np.empty((num_samples, 8), dtype=np.float64)
    for channel in range(8):
        wav[:, channel] = 0.1 * (channel + 1)
    blob = _write(wav, 44100, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    expected_mean = float(np.mean(0.1 * np.arange(1, 9)))
    assert audio.samples.shape[0] == num_samples
    assert np.abs(audio.samples - expected_mean).max() < 1e-3


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


def test_exactly_max_ref_seconds_is_accepted() -> None:
    """9: the boundary itself must decode, not just anything strictly under it."""
    wav = _sine(MAX_REF_SECONDS, _SR)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.shape[0] == MAX_REF_SECONDS * _SR
    assert audio.duration_seconds == pytest.approx(MAX_REF_SECONDS, abs=1e-6)


def test_one_sample_over_max_ref_seconds_is_audio_too_long() -> None:
    """9: the very next sample past the boundary must already be rejected -- the
    bounded read stops at 30 s + 1 sample specifically so this is detectable without
    reading (or trusting the header for) anything longer."""
    num_samples = MAX_REF_SECONDS * _SR + 1
    wav = np.full((num_samples, 1), 0.1, dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "audio_too_long"


def test_zero_samples_is_audio_too_short() -> None:
    """BC-16: a clip too short for even one codec frame must be a 400, not a
    silently-ignored reference. A well-formed header around no sample data at all is
    distinct from the empty-blob and garbage-bytes cases above (both of which fail
    before any header is even parsed)."""
    wav = np.zeros((0, 1), dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"


def test_minimum_clip_length_boundary_at_24khz() -> None:
    """9 / USER DECISION (BC-16): the minimum reference clip is one full 80 ms codec
    frame -- 1,920 samples at the 24 kHz-equivalent (resampled) length. `predicted_frames`
    itself still rounds up (a single sample predicts 1 frame), so this minimum is its
    own, separate check, not derived from `predicted_frames < 1`. At 24 kHz there's no
    resampling, so the boundary is a direct sample count: one sample short is rejected,
    the boundary itself decodes."""
    below = np.full((CODEC_SAMPLES_PER_FRAME - 1, 1), 0.1, dtype=np.float64)
    at_boundary = np.full((CODEC_SAMPLES_PER_FRAME, 1), 0.1, dtype=np.float64)

    with pytest.raises(ApiError) as exc_info:
        decode(_write(below, CODEC_SAMPLE_RATE, format="WAV", subtype="PCM_16"))
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"

    audio = decode(_write(at_boundary, CODEC_SAMPLE_RATE, format="WAV", subtype="PCM_16"))
    assert audio.predicted_frames == 1


def test_minimum_clip_length_boundary_at_44100hz() -> None:
    """9: the same 1,920-sample minimum, but at a rate that actually has to resample
    first. The native sample count at the boundary is found by searching real
    `librosa.resample` output (not the module under test), the same technique
    `test_predicted_frames_matches_actual_librosa_resample` above uses."""
    sample_rate = 44100
    n = 1
    while (
        len(librosa.resample(y=np.zeros(n, dtype=np.float32), orig_sr=sample_rate, target_sr=CODEC_SAMPLE_RATE))
        < CODEC_SAMPLES_PER_FRAME
    ):
        n += 1
    below = np.full((n - 1, 1), 0.1, dtype=np.float64)
    at_boundary = np.full((n, 1), 0.1, dtype=np.float64)

    with pytest.raises(ApiError) as exc_info:
        decode(_write(below, sample_rate, format="WAV", subtype="PCM_16"))
    assert exc_info.value.code == "audio_too_short"

    audio = decode(_write(at_boundary, sample_rate, format="WAV", subtype="PCM_16"))
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
