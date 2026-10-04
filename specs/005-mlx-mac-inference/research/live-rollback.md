# Rollback drill (T031, Constitution VII): 2026-10-04

The user ran the drill on the CUDA machine and reported a pass:
1. They stopped 2.2.0, checked out `v2.1.0`, reinstalled the requirements and restarted with
   `scripts/start_breeze.sh`. `/health` returned 200 within the 5-minute limit.
2. The README `curl` POST and `.wav` examples worked, with `X-Breeze-Version: 2.1.0`.
3. Returning to the branch brought 2.2.0 back the same way.

The timings and outputs were not pasted into this session, so only the user's pass verdict is
recorded here.
