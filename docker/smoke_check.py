"""Import-level validation used while building the public container."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import fastapi
import flash_attn
import python_multipart
import qwen_tts
import starlette
import torch
import transformers
import uvicorn
import websockets

from models.fast_streaming import FastStreamingConfig
from models.warmup_profile import load_warmup_profile

EXPECTED = {
    "torch": "2.9.1",
    "transformers": "4.57.3",
    "qwen-tts": "0.1.1",
    "flash-attn": "2.8.3",
    # Mirrors the exact web-stack pins in requirements.txt.
    "fastapi": "0.141.1",
    "starlette": "1.6.0",
    "uvicorn": "0.52.4",
    "python-multipart": "0.0.32",
}


def main() -> None:
    imported_modules = (
        fastapi, flash_attn, python_multipart, qwen_tts, starlette, torch,
        transformers, uvicorn, websockets,
    )
    if not all(imported_modules):
        raise RuntimeError("one or more required modules failed to import")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        actual = versions[name].split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {versions[name]}")

    cfg = FastStreamingConfig(fast_all=True)
    if not cfg.fast_all:
        raise RuntimeError("fast runtime configuration is unavailable")

    load_warmup_profile(REPO_ROOT / "configs/fast.json")

    print("Container dependency smoke check passed:", versions)


if __name__ == "__main__":
    main()
