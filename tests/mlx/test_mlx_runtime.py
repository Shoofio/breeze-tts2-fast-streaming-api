"""The MLX runtime on the real checkpoint (research R9). Needs BREEZE_MLX_MODEL on Apple Silicon;
token parity also needs BREEZE_MODEL (the official PyTorch checkpoint)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlx

TESTS_DIR = Path(__file__).resolve().parents[1]
# Existing test inputs: segmenter edge cases (CJK, emoji, combining marks, NUL, long runs) and
# template strings. Every string literal in these files is tokenized.
CORPUS_FILES = (TESTS_DIR / "test_text_split.py", TESTS_DIR / "test_templates.py")

# The official checkpoint's config.json values for the fields the server reads.
OFFICIAL_CONFIG = {
    "num_codebooks": 16,
    "codebook_pad_token_id": 2050,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "hidden_size": 2048,
    "head_dim": 128,
}
OFFICIAL_CODEBOOK_SIZE = 2048


def corpus() -> list[str]:
    texts: set[str] = set()
    for path in CORPUS_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
                texts.add(node.value)
    return sorted(texts)


def test_runtime_loads_at_24_khz(mlx_runtime) -> None:
    assert mlx_runtime.sample_rate == 24000


def test_model_config_has_the_official_values(mlx_runtime) -> None:
    config = mlx_runtime.model.config
    for name, value in OFFICIAL_CONFIG.items():
        assert getattr(config, name) == value, name
    assert config.codec_config.codebook_size == OFFICIAL_CODEBOOK_SIZE


def test_tokenizer_matches_the_official_one(mlx_runtime, official_model: Path | None) -> None:
    if official_model is None:
        pytest.skip("set BREEZE_MODEL to the official checkpoint to compare tokenizers")
    from transformers import AutoTokenizer

    official = AutoTokenizer.from_pretrained(official_model, fix_mistral_regex=False)
    texts = corpus()
    assert len(texts) >= 50
    mismatched = [t for t in texts if mlx_runtime.tokenizer(t)["input_ids"] != official(t)["input_ids"]]
    print(f"tokenizer parity: {len(texts)} texts, {len(mismatched)} mismatched")
    assert mismatched == []
