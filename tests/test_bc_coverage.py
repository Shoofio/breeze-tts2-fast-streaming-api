"""SC-002: every breaking change BC-01 to BC-48 has an automated check for the new behavior.

This parses the Breaking Changes table out of `specs/003-cpp-compatible-api/spec.md` (rather than
hard-coding the id range) and the `test_bc_NN_*` functions out of every file under `tests/`
(via `ast`, so nothing is imported or executed), then asserts each BC id has at least one such
test whose docstring states the C++ behavior it rejects (T081). The machine-checkable proxy for
that is the docstring containing "C++": a human still has to write accurate prose, but a test
that never mentions the old behavior can't pass.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "specs" / "003-cpp-compatible-api" / "spec.md"
TESTS_DIR = Path(__file__).resolve().parent

# Matches a Breaking Changes table row: "| BC-07 | Busy check runs before validation | ... |"
BC_ROW = re.compile(r"^\|\s*BC-(\d+)\s*\|\s*(.+?)\s*\|")

# Matches a test function that documents which BC id it covers, e.g. `test_bc_07_...`.
BC_TEST_NAME = re.compile(r"^test_bc_(\d+)_")


def _bc_ids_from_spec() -> dict[str, str]:
    """Return {"BC-07": "<c++ behavior column text>"} for every row in the Breaking Changes table."""
    ids: dict[str, str] = {}
    for line in SPEC_PATH.read_text().splitlines():
        match = BC_ROW.match(line)
        if match:
            number, cpp_behavior = match.groups()
            ids[f"BC-{number}"] = cpp_behavior
    return ids


def _test_functions_by_bc_id() -> dict[str, list[tuple[Path, str, str | None]]]:
    """Return {"BC-07": [(file, test_name, docstring_or_None), ...]} for every test_bc_NN_* function
    under tests/, found by walking the AST (no imports, so this never touches CUDA/GPU fixtures).
    """
    by_bc: dict[str, list[tuple[Path, str, str | None]]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            match = BC_TEST_NAME.match(node.name)
            if not match:
                continue
            bc_id = f"BC-{match.group(1)}"
            docstring = ast.get_docstring(node)
            by_bc.setdefault(bc_id, []).append((path, node.name, docstring))
    return by_bc


def test_every_bc_id_has_a_documented_test() -> None:
    """Each BC-01..BC-48 row in spec.md must have a test_bc_NN_* test whose docstring names "C++"."""
    bc_ids = _bc_ids_from_spec()
    expected = {f"BC-{n:02d}" for n in range(1, 49)}
    # Exact set, so a malformed row can't silently drop an id from the check below.
    assert set(bc_ids) == expected, (
        f"spec.md Breaking Changes table parsed wrongly: missing {sorted(expected - set(bc_ids))}, "
        f"unexpected {sorted(set(bc_ids) - expected)}"
    )

    tests_by_bc = _test_functions_by_bc_id()

    missing_entirely = []
    missing_docstring = []
    for bc_id, cpp_behavior in sorted(bc_ids.items(), key=lambda kv: int(kv[0].split("-")[1])):
        candidates = tests_by_bc.get(bc_id, [])
        if not candidates:
            missing_entirely.append(f"{bc_id} ({cpp_behavior})")
            continue
        if not any(doc and "C++" in doc for _, _, doc in candidates):
            names = ", ".join(f"{p.relative_to(REPO_ROOT)}::{n}" for p, n, _ in candidates)
            missing_docstring.append(f"{bc_id}: {names}")

    problems = []
    if missing_entirely:
        problems.append(
            "no test_bc_NN_* test at all for: " + "; ".join(missing_entirely)
        )
    if missing_docstring:
        problems.append(
            "test(s) exist but none has a docstring naming the C++ behavior (containing \"C++\") for: " + "; ".join(missing_docstring)
        )
    assert not problems, "\n".join(problems)
