"""Regenerate golden.json from the real C++ segmenter.

Compiles harness.cpp against Breeze-TTS-2.cpp's src/text_split.cpp, runs it, checks that the
output parses as JSON, and writes it to golden.json. It writes nothing else:
tests/test_text_split.py reads golden.json at test time and keeps its own table of intentional
differences, so regenerating can never silently rewrite what the tests expect.

Run by hand after editing harness.cpp (see README.md). Needs g++ and a Breeze-TTS-2.cpp checkout.
Not named test_*.py so pytest doesn't collect it.

    .venv/bin/python tests/cpp_golden/gen_goldens.py [--cpp-root /path/to/Breeze-TTS-2.cpp]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_CPP_ROOT = Path("<Breeze-TTS-2.cpp checkout>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpp-root", type=Path, default=DEFAULT_CPP_ROOT)
    root = parser.parse_args().cpp_root

    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "text_split_harness"
        subprocess.run(
            [
                "g++", "-std=c++17",
                f"-I{root / 'include'}",
                f"-I{root / 'third_party/ggml/include'}",
                str(HERE / "harness.cpp"),
                str(root / "src/text_split.cpp"),
                "-o", str(exe),
            ],
            check=True,
        )
        output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout

    json.loads(output)  # refuse to write a golden file that doesn't parse
    (HERE / "golden.json").write_text(output, encoding="utf-8")
    print(f"wrote {HERE / 'golden.json'}")


if __name__ == "__main__":
    main()
