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
import random
import shutil
import subprocess
import time
import warnings

import librosa
import numpy as np
import pytest
import soundfile as sf

from breeze_infer.errors import ApiError
from breeze_infer.limits import MAX_REF_SECONDS
from breeze_infer.reference_audio import decode, predicted_frames
from tests.fakes import CODEC_SAMPLE_RATE, CODEC_SAMPLES_PER_FRAME, codec_frame_count

_SR = 16000

_FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not installed")


def _ffmpeg_flac(source_filter: str, *, frame_size: int | None = None, metadata: dict[str, str] | None = None) -> bytes:
    """A real FLAC produced by piping ffmpeg's own encoder to stdout (``-f flac``
    to ``-``) -- exactly like a client streaming a live capture, rather than
    `_flac_with_bogus_total_samples`'s simulation of one. STREAMINFO's
    ``total_samples`` is left at 0 because ffmpeg can't seek back to fill it in
    once the pipe is closed (confirmed empirically), which is what puts these
    fixtures on the same unknown-length path as the hand-built ones above.
    """
    cmd = [_FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", source_filter]
    if frame_size is not None:
        cmd += ["-frame_size", str(frame_size)]
    for key, value in (metadata or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd += ["-f", "flac", "-y", "-"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


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
    what a streaming FLAC writer emits when the length isn't known upfront (e.g. one
    piped from ``ffmpeg``). libsndfile then reports ``frames`` as ``INT64_MAX``
    (9223372036854775807) rather than raising -- confirmed empirically.

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


def _corrupt_middle_third(blob: bytes, rng: random.Random, num_bytes: int = 64) -> bytes:
    """`num_bytes` random-value bytes at a random offset in the middle third of
    `blob` -- the reproduction that found the silently-truncated-FLAC bug."""
    data = bytearray(blob)
    third = len(data) // 3
    start = third + rng.randrange(third)
    num = min(num_bytes, len(data) - start)
    for i in range(start, start + num):
        data[i] = rng.randrange(256)
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
    """Distinct, unequal channel values prove the downmix actually averages both
    channels -- identical channels would still pass a downmix that silently just took
    one channel and ignored the other."""
    num_samples = int(0.5 * 44100)
    wav = np.empty((num_samples, 2), dtype=np.float64)
    wav[:, 0] = 0.3
    wav[:, 1] = 0.1
    blob = _write(wav, 44100, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.ndim == 1
    assert audio.sample_rate == 44100
    assert audio.samples.shape[0] == num_samples
    # Mean of 0.3 and 0.1 is 0.2; PCM_16 quantizes each channel independently, so the
    # tolerance is a couple of quantization steps (1/32768 each), not exact.
    assert np.abs(audio.samples - 0.2).max() < 1e-4


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
    """A FLAC whose STREAMINFO metadata parses fine (and reports a normal, trustworthy
    length) but whose audio-frame data is cut off mid-stream fails during the read,
    not the open, since the metadata block itself is intact. This confirms the single
    try/except covers both the open and the read, not just the header-only stage."""
    wav = _sine(1.0, _SR)
    blob = _write(wav, _SR, format="FLAC", subtype="PCM_16")
    truncated = blob[: len(blob) // 2]

    with pytest.raises(ApiError) as exc_info:
        decode(truncated)
    assert exc_info.value.code == "invalid_audio"


def test_truncated_ogg_is_invalid_audio() -> None:
    """An OGG truncated mid-stream fails to even open ("Supported file format but
    file is malformed"), the other of the two failure points the combined
    try/except has to cover alongside a FLAC's read-time failure."""
    wav = _sine(1.0, _SR)
    blob = _write(wav, _SR, format="OGG", subtype="VORBIS")
    truncated = blob[: len(blob) // 2]

    with pytest.raises(ApiError) as exc_info:
        decode(truncated)
    assert exc_info.value.code == "invalid_audio"


def test_flac_unknown_length_short_clip_is_accepted() -> None:
    """A FLAC whose STREAMINFO total_samples is left at 0 -- what a streaming encoder
    emits when it doesn't know the length upfront -- decodes normally when the real
    underlying audio is well within the length limit, rather than being rejected just
    because its header can't be trusted to state its own length."""
    wav = _sine(1.0, _SR)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    audio = decode(blob)

    assert audio.sample_rate == _SR
    # The tolerant read can lose part of one internal read block at the true end of
    # an unknown-length stream; a couple of block-widths is a generous bound on that.
    assert audio.duration_seconds == pytest.approx(1.0, abs=0.05)


def test_flac_unknown_length_at_exactly_max_ref_seconds_is_accepted() -> None:
    """The same unknown-length header, with real audio right at the limit: exactly
    MAX_REF_SECONDS of real data is still accepted."""
    wav = _sine(MAX_REF_SECONDS, _SR)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    audio = decode(blob)

    assert audio.duration_seconds == pytest.approx(MAX_REF_SECONDS, abs=0.05)


def test_flac_unknown_length_over_max_ref_seconds_is_audio_too_long() -> None:
    """STREAMINFO total_samples=0 makes libsndfile report frames as an implausible
    sentinel value that must never be trusted for sizing. When the real underlying
    audio genuinely exceeds MAX_REF_SECONDS, the bounded read (capped at 30 s + 1
    sample, regardless of what the header claims) still catches it, without reading
    -- or allocating memory for -- anything like the length the header reports."""
    wav = _sine(MAX_REF_SECONDS + 1, _SR)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_long"
    assert exc_info.value.message == "ref_audio is longer than 30 seconds"


def test_flac_unknown_length_recovers_the_exact_sample_count() -> None:
    """The tolerant read used to lose up to one internal block's worth of the tail;
    it now recovers every sample a read did successfully decode before hitting the
    stream's real end, via a NaN-sentineled output buffer, so a 0.2 s piped clip
    decodes as exactly 0.2 s, not slightly under it."""
    wav = _sine(0.2, _SR)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    audio = decode(blob)

    assert audio.samples.shape[0] == wav.shape[0]
    assert audio.duration_seconds == pytest.approx(0.2, abs=1e-9)


def test_flac_unknown_length_at_exactly_the_minimum_is_accepted() -> None:
    """An 80 ms (1,920-sample) piped clip is exactly the minimum accepted length --
    exact tail recovery must not lose even a single sample of that margin."""
    wav = np.full((1920, 1), 0.1, dtype=np.float64)
    blob = _flac_with_bogus_total_samples(wav, CODEC_SAMPLE_RATE)

    audio = decode(blob)

    assert audio.samples.shape[0] == 1920
    assert audio.predicted_frames == 1


def test_flac_unknown_length_under_minimum_is_audio_too_short_not_invalid_audio() -> None:
    """A piped clip well under the old 256-sample block size (and under the 80 ms
    minimum) must be classified by its actual length -- audio_too_short -- not
    misread as unreadable because it happened to be short."""
    wav = np.full((100, 1), 0.1, dtype=np.float64)
    blob = _flac_with_bogus_total_samples(wav, _SR)

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"


@requires_ffmpeg
def test_flac_unknown_length_with_large_comment_is_accepted() -> None:
    """A large comment block is real metadata, not audio, and must not affect
    whether the recording decodes -- regression coverage for a heuristic that used
    to size-check the whole file (comment included) and wrongly rejected this."""
    blob = _ffmpeg_flac(
        "sine=frequency=220:duration=2:sample_rate=24000",
        metadata={"comment": "x" * 120_000},
    )

    audio = decode(blob)

    assert audio.sample_rate == 24000
    assert audio.duration_seconds == pytest.approx(2.0, abs=1e-6)


@requires_ffmpeg
def test_flac_unknown_length_with_padding_and_tags_is_accepted() -> None:
    """ffmpeg's own FLAC muxer adds a padding block by default alongside a Vorbis
    comment block for title/artist/album -- real metadata that must not affect
    whether the recording decodes. Uses noise (seeded for a reproducible fixture)
    at the smallest allowed sample rate."""
    blob = _ffmpeg_flac(
        "anoisesrc=color=white:duration=0.08:sample_rate=8000:seed=1",
        metadata={"title": "T", "artist": "A", "album": "Alb"},
    )

    audio = decode(blob)

    assert audio.sample_rate == 8000
    assert audio.duration_seconds == pytest.approx(0.08, abs=1e-6)


@requires_ffmpeg
@pytest.mark.parametrize("frame_size", [16, 32])
def test_flac_unknown_length_with_small_frame_size_is_accepted(frame_size: int) -> None:
    """Many small frames (ffmpeg's -frame_size) is still a perfectly ordinary
    stream, decodable start to finish -- regression coverage for a heuristic that
    used to size-check against a per-frame overhead estimate and wrongly rejected
    this when there were many more (smaller) frames than it assumed."""
    blob = _ffmpeg_flac("sine=frequency=220:duration=1:sample_rate=24000", frame_size=frame_size)

    audio = decode(blob)

    assert audio.sample_rate == 24000
    assert audio.duration_seconds == pytest.approx(1.0, abs=1e-6)


@requires_ffmpeg
@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000])
@pytest.mark.parametrize("duration", [0.15, 0.3, 0.5])
def test_flac_unknown_length_with_short_final_frame_is_accepted(duration: float, sample_rate: int) -> None:
    """The specific case review found still wrongly rejected: a short clip encoded
    with a large -frame_size (4608) ends in one short, partial final frame relative
    to the rest -- exactly what made the old per-frame-overhead heuristic misjudge
    the file's plausible size. Seeded noise (#4) for a reproducible fixture, across
    every allowed sample rate at the low end where this bit hardest."""
    blob = _ffmpeg_flac(f"anoisesrc=color=white:duration={duration}:sample_rate={sample_rate}:seed=1", frame_size=4608)

    audio = decode(blob)

    assert audio.sample_rate == sample_rate
    assert audio.duration_seconds == pytest.approx(duration, abs=1e-3)


@requires_ffmpeg
def test_flac_unknown_length_under_minimum_wins_over_end_of_stream_check() -> None:
    """4: too-short is checked before the end-of-stream check, so a piped clip that
    ends up under 80 ms is always audio_too_short -- never invalid_audio."""
    blob = _ffmpeg_flac("sine=frequency=220:duration=0.05:sample_rate=16000")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"


@requires_ffmpeg
def test_flac_unknown_length_tail_cut_is_accepted_with_fewer_samples() -> None:
    """Known limit (module docstring, point 3): a piped FLAC cut short can decode
    successfully with fewer samples than the original, when the cut happens to
    land somewhere libsndfile's own bookkeeping treats as a plausible end rather
    than a decode error. This isn't a corruption bypass -- `_corrupt_middle_third`
    above still gets reliably rejected by libsndfile itself -- it's a real gap:
    nothing here can tell a deliberately shortened but well-formed stream apart
    from a genuinely short recording without an independently known length. A
    small sweep of exact cut points finds one (some cut points instead hit a
    genuine decode error and are rejected, which is also fine -- the point is that
    at least one accepted, shortened case exists, documenting the gap rather than
    asserting it never happens)."""
    clean_blob = _ffmpeg_flac("sine=frequency=220:duration=1:sample_rate=16000")
    clean_audio = decode(clean_blob)

    found_shorter_accept = False
    for lost_bytes in range(1, 500):
        truncated = clean_blob[: len(clean_blob) - lost_bytes]
        try:
            audio = decode(truncated)
        except ApiError:
            continue
        assert audio.samples.shape[0] <= clean_audio.samples.shape[0]
        if audio.samples.shape[0] < clean_audio.samples.shape[0]:
            found_shorter_accept = True
            break

    assert found_shorter_accept


@requires_ffmpeg
def test_flac_unknown_length_never_silently_accepts_truncated_corruption() -> None:
    """Randomized mid-stream corruption (64 bytes replaced at a random offset in the
    middle third of the file) must never produce a silently truncated "success".
    libFLAC is genuinely resilient to a lot of random corruption -- many trials
    decode with output identical to the clean file, since the corrupted bytes
    happened to land somewhere the decoder tolerates -- but every trial that
    doesn't decode losslessly must be rejected outright, never accepted with a
    plausible-looking but shorter or otherwise different result. That silent
    truncation, not "any corruption at all", was the actual bug: an accepted
    result's samples, not just its length or duration, must always match the clean
    file exactly. This rejection comes from libsndfile's own decode failure on the
    corrupted data (a different, non-tolerated error from the one that signals a
    genuine end-of-stream), not from any check of ours -- confirmed by the fact
    that removing the (since-removed) size heuristic didn't change this test's
    outcome. Uses a real ffmpeg-piped fixture, not the hand-built simulation.
    """
    clean_blob = _ffmpeg_flac("sine=frequency=220:duration=1:sample_rate=16000")
    clean_audio = decode(clean_blob)

    rng = random.Random(0)
    accepted = rejected = 0
    for _ in range(150):
        corrupted = _corrupt_middle_third(clean_blob, rng)
        try:
            audio = decode(corrupted)
        except ApiError as exc:
            assert exc.code == "invalid_audio"
            rejected += 1
            continue
        accepted += 1
        # Accepted only because this particular corruption changed nothing
        # decodable -- never a truncated or altered fraction of the real content.
        assert np.array_equal(audio.samples, clean_audio.samples)

    assert accepted + rejected == 150
    assert rejected > 0  # this corruption technique does bite sometimes


def test_flac_unknown_length_decodes_quickly_even_when_incompressible() -> None:
    """Performance sanity bound: a 192 kHz stereo clip near the 30 s cap, filled with
    incompressible noise (so the file is genuinely large, comparable to a real
    17 MiB reproduction) must still decode in well under a second -- large blocks
    are the common read path for an unknown-length stream too, not just a
    trustworthy one, and the exact-recovery machinery only runs once, at the tail.
    """
    rng = np.random.default_rng(0)
    sample_rate = 192_000
    num_samples = int(29.9 * sample_rate)
    wav = rng.uniform(-0.9, 0.9, (num_samples, 2))
    blob = _flac_with_bogus_total_samples(wav, sample_rate)

    start = time.monotonic()
    audio = decode(blob)
    elapsed = time.monotonic() - start

    assert audio.samples.shape[0] == num_samples
    assert elapsed < 3.0  # generous bound; measured well under 1 s locally


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_samples_is_invalid_audio(bad_value: float) -> None:
    """A float WAV can carry NaN/Inf verbatim -- soundfile doesn't reject it on write
    or read -- so decode() must reject it, rather than handing the codec non-finite
    input."""
    wav = _sine(0.1, _SR)
    wav[len(wav) // 2, 0] = bad_value
    blob = _write(wav, _SR, format="WAV", subtype="FLOAT")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_audio"
    assert exc_info.value.message == "could not read ref_audio"


def test_absurdly_large_sample_is_invalid_audio() -> None:
    """A finite but wildly out-of-range sample (real reference audio stays within
    +/-1.0) would overflow arithmetic downstream in the codec, so it's rejected here
    instead, the same as a non-finite one."""
    wav = _sine(0.1, _SR)
    wav[len(wav) // 2, 0] = 100.0
    blob = _write(wav, _SR, format="WAV", subtype="FLOAT")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_audio"
    assert exc_info.value.message == "could not read ref_audio"


def test_near_float32_max_samples_on_two_channels_is_invalid_audio_without_warning() -> None:
    """Two channels each near float32's ~3.4e38 max would overflow a naive float32
    sum during the downmix before any magnitude check could even run -- the decode
    must accumulate in float64 instead, so this is rejected cleanly (as absurdly
    large, not as non-finite) and raises no RuntimeWarning."""
    num_samples = int(0.1 * _SR)
    wav = np.full((num_samples, 2), 3e38, dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="FLOAT")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ApiError) as exc_info:
            decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_audio"
    assert exc_info.value.message == "could not read ref_audio"


def test_reads_across_internal_block_boundaries_correctly() -> None:
    """The decode loop reads in bounded blocks rather than one large sf.read call, so
    a clip spanning several block boundaries -- at the max channel count, where a
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
    """The boundary itself must decode, not just anything strictly under it."""
    wav = _sine(MAX_REF_SECONDS, _SR)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    audio = decode(blob)

    assert audio.samples.shape[0] == MAX_REF_SECONDS * _SR
    assert audio.duration_seconds == pytest.approx(MAX_REF_SECONDS, abs=1e-6)


def test_one_sample_over_max_ref_seconds_is_audio_too_long() -> None:
    """The very next sample past the boundary must already be rejected -- the bounded
    read stops at 30 s + 1 sample specifically so this is detectable without reading
    (or trusting the header for) anything longer."""
    num_samples = MAX_REF_SECONDS * _SR + 1
    wav = np.full((num_samples, 1), 0.1, dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.code == "audio_too_long"


def test_zero_samples_is_audio_too_short() -> None:
    """BC-16: a clip too short for even one codec frame must be a 400, not a
    silently-ignored reference. A well-formed header around no sample data at all is
    distinct from an empty upload or garbage bytes, both of which fail before any
    header is even parsed."""
    wav = np.zeros((0, 1), dtype=np.float64)
    blob = _write(wav, _SR, format="WAV", subtype="PCM_16")

    with pytest.raises(ApiError) as exc_info:
        decode(blob)
    assert exc_info.value.status == 400
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"


def test_minimum_clip_length_boundary_at_24khz() -> None:
    """BC-16 (spec decision): the minimum reference clip is exactly 80 ms, checked in
    integer arithmetic on the native sample count -- reject when
    ``n_samples * 24000 < 1920 * sample_rate`` -- rather than through the (rounding)
    resample step: `predicted_frames` alone can't express this minimum, since its two
    nested ceiling divisions mean it never reports fewer than 1 frame for any nonzero
    input. At 24 kHz the check reduces to a direct sample count: one sample short of
    1,920 is rejected, 1,920 itself decodes."""
    below = np.full((1919, 1), 0.1, dtype=np.float64)
    at_boundary = np.full((1920, 1), 0.1, dtype=np.float64)

    with pytest.raises(ApiError) as exc_info:
        decode(_write(below, CODEC_SAMPLE_RATE, format="WAV", subtype="PCM_16"))
    assert exc_info.value.code == "audio_too_short"
    assert exc_info.value.message == "ref_audio is too short"

    audio = decode(_write(at_boundary, CODEC_SAMPLE_RATE, format="WAV", subtype="PCM_16"))
    assert audio.samples.shape[0] == 1920


def test_minimum_clip_length_boundary_at_44100hz() -> None:
    """The same exact boundary at a rate where the cross-multiplication actually
    matters: ``3528 * 24000 == 1920 * 44100`` exactly, so 3,527 samples is rejected
    and 3,528 decodes."""
    below = np.full((3527, 1), 0.1, dtype=np.float64)
    at_boundary = np.full((3528, 1), 0.1, dtype=np.float64)

    with pytest.raises(ApiError) as exc_info:
        decode(_write(below, 44100, format="WAV", subtype="PCM_16"))
    assert exc_info.value.code == "audio_too_short"

    audio = decode(_write(at_boundary, 44100, format="WAV", subtype="PCM_16"))
    assert audio.samples.shape[0] == 3528


def test_minimum_clip_length_boundary_at_192khz() -> None:
    """The same boundary at the maximum allowed sample rate:
    ``15360 * 24000 == 1920 * 192000`` exactly, so 15,359 samples is rejected and
    15,360 decodes."""
    below = np.full((15359, 1), 0.1, dtype=np.float64)
    at_boundary = np.full((15360, 1), 0.1, dtype=np.float64)

    with pytest.raises(ApiError) as exc_info:
        decode(_write(below, 192000, format="WAV", subtype="PCM_16"))
    assert exc_info.value.code == "audio_too_short"

    audio = decode(_write(at_boundary, 192000, format="WAV", subtype="PCM_16"))
    assert audio.samples.shape[0] == 15360


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


@pytest.mark.parametrize(
    "sample_rate", [8_000, 11_025, 16_000, 22_050, 24_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000]
)
def test_max_ref_frames_covers_a_full_length_reference_at_every_rate(sample_rate: int) -> None:
    """voice_file bounds a saved voice's frames by MAX_REF_FRAMES; a 30 s reference at
    any accepted rate must fit under it."""
    from breeze_infer.reference_audio import MAX_REF_FRAMES

    assert predicted_frames(30 * sample_rate, sample_rate) <= MAX_REF_FRAMES


def test_max_ref_frames_is_reached_by_float_rounding_at_some_rates() -> None:
    """Why MAX_REF_FRAMES is 376, not 30 s x 12.5 fps = 375: at 191,995 Hz the
    resampler's float ratio pushes a full 30 s one sample past 720,000."""
    from breeze_infer.reference_audio import MAX_REF_FRAMES

    assert MAX_REF_FRAMES == 376
    assert predicted_frames(30 * 191_995, 191_995) == MAX_REF_FRAMES
