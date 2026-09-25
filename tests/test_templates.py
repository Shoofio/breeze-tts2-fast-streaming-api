from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from breeze_infer.templates import _encode_prompt_audio
from tests.fakes import FakeCodec, codec_frame_count


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, list[int] | torch.Tensor]:
        del add_special_tokens
        ids = list(range(2, 2 + len(text)))
        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, input_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "x" * len(input_ids)


def test_encode_prompt_audio_reads_with_soundfile_and_downmixes(tmp_path) -> None:
    audio_path = tmp_path / "stereo.wav"
    wav = np.stack(
        [
            np.linspace(-0.5, 0.5, 8, dtype=np.float32),
            np.linspace(0.5, -0.5, 8, dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(audio_path, wav, 24000)
    tokenizer = FakeCodec()

    codes = _encode_prompt_audio(tokenizer, str(audio_path))

    assert isinstance(codes, torch.Tensor)
    # 8 samples at 24 kHz is a fraction of one 1920-sample codec frame, so this is 1 frame,
    # not the old fake's hard-coded 4 -- ``codec_frame_count`` is the real formula (also
    # exercised directly in tests/test_fakes.py).
    assert tuple(codes.shape) == (codec_frame_count(8, 24000), 16)
    assert tokenizer.last_sr == 24000
    assert tokenizer.last_wav is not None
    assert tokenizer.last_wav.shape == (8,)
    np.testing.assert_allclose(tokenizer.last_wav, np.mean(wav, axis=1), atol=1e-4)
