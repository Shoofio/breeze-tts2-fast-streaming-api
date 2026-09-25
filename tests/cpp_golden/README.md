# C++ golden harness for `text_split.py`

`breeze_infer/text_split.py` is a Python port of Breeze-TTS-2.cpp's `src/text_split.cpp`
(`split_text`) and of `apps/server/ws_api.cpp`'s `sentence_end`/`drain`, with the fixes in
`specs/003-cpp-compatible-api/research.md` R11. Rather than hand-deriving what the C++ does,
`tests/test_text_split.py` compares against what the real C++ produces, captured here.

Nothing here is compiled at test time. The test reads `golden.json`; `g++` is needed only when
you change `harness.cpp` and regenerate it.

## Files

- `harness.cpp` links the real `text_split.cpp` for `split_text` and embeds a verbatim copy of
  `ws_api.cpp`'s `sentence_end`/`drain`, so it exercises the server's drain without its
  networking code. It prints one JSON object, `{"split_text": [...], "drain": [...]}`, and
  each case carries its own inputs next to its result.
- `golden.json` is the harness's output, committed so the tests need no compiler.
- `gen_goldens.py` compiles and runs the harness and writes `golden.json`. It writes nothing
  else. Not named `test_*.py`, so pytest doesn't collect it.

## Regenerating

After adding or editing a case in `harness.cpp`'s `split_cases`/`drain_cases`:

```sh
.venv/bin/python tests/cpp_golden/gen_goldens.py --cpp-root <Breeze-TTS-2.cpp checkout>
```

Then run `.venv/bin/pytest -q tests/test_text_split.py`. A new case that the Python port
intentionally answers differently fails until you add it to `INTENTIONAL_DIFFERENCES` in the
test file, with the expected output and its BC id.

## How the test uses `golden.json`

- `split_text` entries (`name`, `text`, `budget`, `first_budget`, `result`) are compared with
  `split_text(text, budget=..., first_budget=...)`. A `first_budget` of 0 means unset, as in C++.
- `drain` entries (`name`, `buffer`, `budget`, `force`, `pieces`, `remaining`) are compared with
  `segment(buffer, budget=..., final=force)`.
- The C++ pieces are normalized first: stripped of surrounding whitespace, empty ones dropped.
  The port always does this (R11), so it isn't counted as a difference.
- Every case whose output still differs after that is listed in `INTENTIONAL_DIFFERENCES` with
  its new expected output and a BC id. The test also fails if an entry there no longer differs
  from the C++, so the table can't go stale.
