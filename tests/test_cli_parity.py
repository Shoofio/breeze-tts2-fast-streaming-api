"""The CLI (`infer.py`) and the server load the model with the same limits.

`infer.py` keeps its own copies of the limits so it stays a standalone script. This test
catches the two drifting apart. The constants are read with `ast` rather than by importing
`infer.py`: the import pulls in the whole streaming runtime, and nothing here needs it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from breeze_infer import limits, model_loading

INFER_PY = Path(__file__).resolve().parent.parent / "infer.py"


def _module_constants(path: Path) -> dict[str, object]:
    """Top-level `NAME = <literal>` assignments in `path`."""
    constants: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue  # not a literal; not one of ours
    return constants


def test_cli_max_new_tokens_matches_the_server_ceiling() -> None:
    assert _module_constants(INFER_PY)["MAX_NEW_TOKENS"] == limits.MAX_NEW_TOKENS_CEILING


def test_cli_max_seq_len_matches_the_server_model_load() -> None:
    assert _module_constants(INFER_PY)["MAX_SEQ_LEN"] == model_loading.MAX_SEQ_LEN
