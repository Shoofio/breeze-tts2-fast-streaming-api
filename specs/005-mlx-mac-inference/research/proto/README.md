# Prototype frame loop (reference only, not product code)

These are the throwaway scripts behind [../proto-2026-10-04.md](../proto-2026-10-04.md) and the
re-gate in [../live-phase0.md](../live-phase0.md). They are kept as the reference for tasks
T016–T018. They are not imported by the server, not tested, and not maintained. Delete this
directory once `models/mlx_streaming.py` has replaced them.

- `proto.py`: the frame loop at levels stock and a–e: depth KV cache, CFG as batch 2, no
  per-token syncs, `mx.compile`, codec on a second stream. Run it as
  `python proto.py 8bit passage stock,e [--cfg]` from this directory.
- `common.py`: model loading, the benchmark texts, and WAV output.
- `diag_teacher.py`: the teacher-forced logit comparison and its tie rule, which T018's test
  reuses.

Environment: Python 3.12 with mlx-audio at `e1b19b9054bf163f5d812221a54fcc346f1890e9`, plus the
Mac overrides (research R2). Weights come from the pinned snapshots in research R3.
