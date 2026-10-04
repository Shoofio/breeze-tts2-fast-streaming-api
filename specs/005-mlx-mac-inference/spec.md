# Feature Specification: MLX Inference on Apple Silicon Macs

**Feature Branch**: `005-mlx-mac-inference`

**Created**: 2026-10-03

**Status**: Draft

**Input**: User description: "I would like to expand this server to support MLX and macs for inference"

**Context found while specifying**:

- Today the server runs only on an NVIDIA GPU with CUDA. Device choice, model loading, the
  single-generation gate, warmup and every `--fast-*` option assume CUDA. About 146 CUDA
  references sit across 21 files in `breeze_infer/` and `models/`.
- Community MLX conversions of Breeze TTS 2 already exist on Hugging Face, all unofficial:
  `mlx-community/Breeze-TTS-2-mlx` (bf16, plus 8-bit and 4-bit variants),
  `rishikksh20/Breeze-TTS-2-mlx` (INT8), and `LunaFox/Breeze-TTS-2-mlx-4bit`. MLX runtimes for
  Qwen3-TTS, the codec this model uses, also exist. Whether any of them can stream audio, or
  supports voice direction and classifier-free guidance, has not been checked. That is
  plan-phase research.
- The reference machine is the development Mac: Apple M5, 16 GB unified memory.

## Clarifications

### Session 2026-10-03

- Q: What does the first release cover? → A: **Full parity of the server**: every HTTP route,
  the WebSocket API, and all voice modes (User Stories 1–3). `infer.py` was later moved out of
  scope (see the next session).
- Q: Where do the Mac weights come from? → A: **An existing community MLX conversion** from
  Hugging Face. The repository does not write its own converter.
- Q: Which precisions? → A: **bf16 and 8-bit.** 4-bit is out of scope.
- Q: Why not PyTorch MPS? → A: **The user tried it.** The existing PyTorch code runs on MPS, but
  it takes more than 5 seconds to generate each second of audio, which is too slow to stream.

### Session 2026-10-03 (2)

- Q: Is command-line synthesis (`infer.py`) on a Mac part of the first release? → A: **No.** The
  first release covers the server only. `infer.py` keeps its current behavior on every
  platform, and a later release may add the Mac backend to it.
- Q: If a voice saved on Linux can't be loaded unchanged by the Mac server, should the Mac server
  still accept it? → A: **No. Voices don't need to move between backends.** Each machine keeps
  its own voices directory. A voice file whose codec fingerprint doesn't match the running
  backend is skipped, as the server already does today.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Streamed speech from a Mac (Priority: P1)

A user with an Apple Silicon Mac and no NVIDIA GPU clones the repository, downloads the model and
starts the server with one command. A client (curl, SillyTavern, a browser `<audio>` element)
sends the same requests it would send to the Linux server and gets the same audio format back.
Playback starts while synthesis is still running.

**Why this priority**: This is the feature. Without it, Mac users can't run the server at all.

**Independent Test**: On the reference Mac, start the server with the macOS launcher. Then run
the README's "First request" examples (`POST /v1/audio/speech` and
`GET /v1/audio/speech.wav`) unchanged. Both return playable 24 kHz mono speech, and audio
arrives before generation finishes.

**Acceptance Scenarios**:

1. **Given** an Apple Silicon Mac with the model downloaded, **When** the user runs the macOS
   launcher, **Then** the server reaches ready (`/health` returns `200`) with no NVIDIA
   hardware or CUDA software present.
2. **Given** a ready Mac server, **When** a client posts text to `/v1/audio/speech`, **Then**
   the server streams 24 kHz mono s16le PCM, and the first bytes arrive before generation
   completes.
3. **Given** a ready Mac server, **When** a browser opens a `/v1/audio/speech.wav` URL,
   **Then** it plays the progressive WAV stream as it does from the Linux server.
4. **Given** a generation in progress, **When** a second `POST /v1/audio/speech` arrives,
   **Then** it gets `409 busy`, as on the Linux server.
5. **Given** a generation in progress, **When** the client disconnects, **Then** generation
   stops and the next request can start.

---

### User Story 2 - Voice features on a Mac (Priority: P2)

A user on the Mac server uses voice clone (reference audio plus transcript), voice design (a
description with no reference), voice direction (reference plus instruction), and saved voices
through `/v1/voices`. Voices saved on the Mac belong to the Mac server. They don't need to work
on a Linux server, and Linux-saved voices don't need to work on the Mac.

**Why this priority**: These are the model's headline capabilities and the fork's voice
management. Without them the Mac server is a plain default-voice reader.

**Independent Test**: On the Mac, save a voice from reference audio. Synthesize with it, with a
design instruction, and with a direction instruction. Restart the server and synthesize with the
same `voice_id` again.

**Acceptance Scenarios**:

1. **Given** a ready Mac server, **When** a client uploads a voice through `/v1/voices` and then
   requests speech with its `voice_id`, **Then** the output carries the reference speaker's
   identity.
2. **Given** a ready Mac server, **When** a client requests speech with an instruction and no
   reference, **Then** the voice follows the description.
3. **Given** voices saved on the Mac, **When** the Mac server restarts, **Then** every saved
   voice is listed and usable without re-uploading.
4. **Given** a voices directory written by the Linux server, **When** the Mac server starts with
   it, **Then** each voice whose codec fingerprint doesn't match is skipped and reported in a
   structured event. The server still starts and serves the voices that do match.

---

### User Story 3 - WebSocket streaming on a Mac (Priority: P3)

A client that uses the WebSocket API (port 8081 by default) connects to the Mac server and
streams speech with the same messages as on Linux.

**Why this priority**: Some clients use only WebSocket, but the HTTP routes cover the main use
case.

**Independent Test**: Run the existing live checks that use the WebSocket (the SillyTavern
`full` run and the C++ docs example check) against the Mac server unchanged.

**Acceptance Scenarios**:

1. **Given** a ready Mac server, **When** a client opens a WebSocket session and sends text,
   **Then** it receives audio frames and completion messages in the documented order.

---

### User Story 4 - Linux and Windows users are unaffected (Priority: P1)

An existing user on Linux, WSL or Windows updates to the release that includes Mac support. The
setup, launch flags, performance and behavior they rely on stay the same.

**Why this priority**: The CUDA path is the fork's working product and is used daily. A Mac
backend that degrades it would be a net loss.

**Independent Test**: On the CUDA machine, run the GPU test suite and `bench_api`, and compare
the results with the 2.1.0 baseline.

**Acceptance Scenarios**:

1. **Given** a Linux CUDA machine, **When** the user installs from `requirements.txt` and runs
   `scripts/start_breeze.sh`, **Then** nothing Mac-specific is installed, and the server
   starts with the same defaults as before.
2. **Given** the CUDA server, **When** the benchmark runs, **Then** time-to-first-audio and
   throughput match the 2.1.0 baseline within measurement noise.

---

### Edge Cases

- **Intel Mac, or a Mac with too little memory**: the server refuses to start, before it loads
  any weights, and says the machine needs Apple Silicon or names the memory it needs.
- **The Mac backend requested on Linux or Windows, or the CUDA backend requested on a Mac**: the
  server refuses to start with a message naming the backends that platform supports.
- **CUDA-only options on a Mac** (`--fast-all`, the other `--fast-*` flags,
  `--attn-implementation`, `--compile-cache-dir`): the server refuses to start and names the
  unsupported option. It does not silently ignore it (Constitution X: no silent fallbacks).
- **Mac-format weights not downloaded, or only the CUDA checkpoint present**: the server refuses
  to start and prints the exact download command.
- **Generation slower than playback on a smaller Mac**: audio still streams. The client may run
  out of buffered audio and pause (underrun). The server neither errors nor truncates the audio.
  The docs give the measured speed for the reference Mac.
- **Long text that needs splitting**: it is split and stitched exactly as on the CUDA backend,
  since text splitting does not depend on the backend.
- **The Mac sleeps or the process is suspended during a generation**: on resume the request
  either completes or fails with the existing error responses. The single-generation gate does
  not stay held.
- **A voice saved by the other backend**: if its codec fingerprint doesn't match, it is skipped
  at startup with the existing skip event, and the server starts normally. If a client requests
  it, the client gets `404 unknown_voice`, as for any voice that doesn't exist.
- **Voices saved before this feature on Linux**: they still load on Linux, with no migration.

## Requirements *(mandatory)*

### Functional Requirements

**Platform and backend selection**

- **FR-001**: The server MUST run inference on Apple Silicon Macs (M1 or later) under macOS,
  using MLX, with no NVIDIA GPU or CUDA software present.
- **FR-002**: The server MUST pick the inference backend once, at startup. On macOS arm64 the
  default is MLX; on every other platform it is CUDA, as today. An explicit launch option MAY
  override the default, and an override the platform can't satisfy MUST stop startup with an
  actionable message.
- **FR-003**: On a platform it can't serve (an Intel Mac, a Mac below the memory minimum), the
  server MUST refuse to start before loading weights, and the message MUST name the
  requirement that isn't met.
- **FR-004**: Launch options that only apply to the CUDA backend MUST stop startup with a message
  naming each option, when given to the Mac backend.

**API compatibility**

- **FR-005**: The Mac backend MUST serve the HTTP and WebSocket API documented in `docs/api.md`
  unchanged: the same routes, request fields, validation, error codes and bodies, audio format
  (24 kHz mono s16le PCM; progressive WAV on the `.wav` route), and `X-Breeze-Version` header.
  The first release MUST cover all of it (User Stories 1–3). The limit on how much text fits in
  one request ("room") follows CUDA's exact-length rule. With the CUDA launcher's `--fast-all`,
  CUDA pads prompts to 32-token buckets, so the Mac backend can accept up to 31 more frames
  before returning `400 text_too_long` (research R4).
- **FR-005a**: `infer.py` is out of scope for this release. Its behavior and options MUST NOT
  change on any platform, and the README MUST say that the Mac backend is available through the
  server only.
- **FR-006**: The Mac backend MUST stream audio: the first audio MUST reach the client before
  generation finishes, on every streaming route it serves.
- **FR-007**: The Mac backend MUST keep the server's concurrency behavior: one generation at a
  time, `409 busy` on `POST /v1/audio/speech` while busy, and the 60-second bounded queue then
  `503 busy_timeout` on the `.wav` route.
- **FR-008**: A client disconnect or abort MUST stop generation on the Mac backend and free it
  for the next request, as on the CUDA backend.
- **FR-009**: Voice clone, voice design, voice direction, `cfg_scale`, `seed` and the sampling
  fields MUST behave on the Mac backend as `docs/api.md` documents. Output is not expected to
  be identical to CUDA output, since numerics and random streams differ between backends. It
  must be comparable in quality and speaker identity (SC-005).
- **FR-010**: Saved voices are NOT required to move between backends. The Mac backend MUST save
  voices with its own codec fingerprint and MUST load voices whose fingerprint matches its
  codec. Voices whose fingerprint doesn't match MUST be skipped at startup, using the existing
  skip-and-report rule, and MUST NOT stop the server starting. Voice files on the CUDA backend
  MUST keep their current format and fingerprint, so existing voices keep working there.

**Model weights**

- **FR-011**: The Mac backend MUST load an existing community MLX conversion of Breeze TTS 2,
  downloaded from Hugging Face. The repository MUST NOT include its own weight converter. The
  docs and launcher MUST pin the conversion to a specific repository and revision, so that every
  install gets the same weights. The README MUST say that the conversion is unofficial and
  unaffiliated with BreezeBlue, and that the BreezeBlue non-commercial licence still applies.
- **FR-012**: The Mac backend MUST support bf16 and 8-bit weights, chosen at launch, with bf16
  as the default. 4-bit weights are out of scope. Choosing them MUST stop startup with a message
  that names the supported precisions.
- **FR-013**: When the weights are missing, the macOS launcher MUST refuse to start and print the
  exact command that downloads them. When the server is started directly with a checkpoint
  directory of the wrong format, it MUST refuse with the download command (contracts:
  launch-and-events). A missing directory keeps the existing "must be an existing directory"
  error.

**Observability and operation**

- **FR-014**: The `model.loaded` startup event MUST report the backend, the device and the
  weight variant. `/health` MAY add these as new fields. Its existing fields and status codes
  MUST NOT change (Constitution IV: additive within a version).
- **FR-015**: A macOS launcher MUST start the server with one command, find the model in the
  Hugging Face cache (`$HF_HOME`) as the Linux launcher does, and use the same binding and CORS
  defaults as the other launchers.
- **FR-016**: Mac-only dependencies MUST be installed only on macOS. Installing on Linux or
  Windows MUST NOT pull them in, and installing on a Mac MUST NOT require CUDA packages.

**No regression on CUDA**

- **FR-017**: On Linux, WSL and Windows, launch options, defaults and documented behavior MUST be
  unchanged. The existing CPU test suite and GPU test suite MUST pass. Existing tests may change
  only to correct assumptions that are true on Linux but not on macOS (research R10). Each such
  test MUST still prove the same thing on Linux, and no assertion about CUDA behaviour may be
  relaxed.
- **FR-018**: The Mac backend MUST have its own test marker, analogous to `gpu`. It runs the
  model-backed tests on a Mac and skips them elsewhere. The model-free tests (`.venv/bin/pytest`)
  MUST pass on macOS.
- **FR-019**: The README and `docs/api.md` MUST document the Mac requirements, quick start,
  supported options, the options that are unavailable, and the measured performance on the
  reference Mac.

### Key Entities

- **Inference backend**: the engine that runs the model on a device: CUDA, which exists today,
  or MLX, which is new. It is chosen once at startup. It owns model loading, warmup, generation
  and aborts. It never changes the API's request or response shapes.
- **Model weights variant**: the checkpoint one backend loads. It has a format (the PyTorch
  checkpoint for CUDA, MLX weights for Mac), a precision (bf16 or quantized), and a source
  (official or a community conversion).
- **Saved voice**: the transcript, the codec tokens encoded from the reference audio, and a
  fingerprint of the codec that encoded them, stored in the voices directory. A voice belongs to
  the backend whose codec fingerprint it carries. Per-backend derived data (cached voice
  prefixes) lives in memory only.
- **Startup report**: the facts in the `model.loaded` event: backend, device, weights variant,
  and warmup time.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the reference Mac, with the model already downloaded, a new user goes from a
  fresh clone to hearing the first request's audio in under 10 minutes, following only the
  README.
- **SC-002**: On the reference Mac, a one-sentence request with a saved voice produces its first
  audio in under 2 seconds. Over a passage of about one minute, the server produces audio at
  least as fast as it plays (no underrun at 1× playback).
- **SC-002a**: 8-bit meets SC-002 on the reference Mac. bf16 must also meet it, unless the
  measurements in the Phase-0 gate and the performance gate show the 16 GB reference Mac
  can't. In that case the docs
  recommend 8-bit for 16 GB Macs and give the measured bf16 numbers.
- **SC-003**: The existing live acceptance checks (the SillyTavern `full` run and the C++ docs
  example check, both of which exercise the WebSocket API too) pass against the Mac server with
  no changes to the checks.
- **SC-004**: On the CUDA machine, `bench_api` time-to-first-audio and throughput stay within 5%
  of the 2.1.0 baseline, and the GPU test suite passes.
- **SC-005**: For 10 fixed prompts that cover clone, design and direction, a listener judges each
  Mac output intelligible, free of artifacts, and matching the CUDA output's speaker identity or
  described voice in all 10. This holds at both bf16 and 8-bit.
- **SC-006**: On the reference Mac (16 GB), the server's peak memory during sustained use stays
  low enough that the system does not swap with a browser and editor open. The performance gate
  sets the number from measurement.
- **SC-007**: Every unsupported-platform and unsupported-option case in Edge Cases ends in a
  startup refusal whose message names the cause. None of them produces a crash trace or a
  silent fallback.

## Assumptions

- **MLX, not PyTorch MPS.** The user ran the existing PyTorch code on MPS. It works, but it takes
  more than 5 seconds to generate each second of audio, which is too slow to stream. MPS is out
  of scope. If plan-phase research shows MLX can't meet FR-006 (streaming) either, the spec will
  be revisited.
- **Default weights source.** `mlx-community/Breeze-TTS-2-mlx` is the expected source, because
  it publishes both bf16 and 8-bit. The plan phase confirms that it matches the official
  checkpoint's behavior (all three voice modes, CFG) and picks the revision to pin. If it fails
  that check, the plan stops and comes back to the user. It does not switch to writing a
  converter.
- **Third-party runtime code.** If a community MLX runtime is adopted rather than only its
  weights, it is a new dependency. The plan must justify it and name the alternatives.
- Apple Silicon only. Intel Macs are not supported. Minimum macOS version: whatever the chosen
  MLX release requires.
- The Mac backend has its own performance profile. The CUDA `--fast-*` options have no Mac
  equivalent. Any Mac speed-up options are designed in the plan phase and must earn their place
  (Constitution II).
- The server stays a single process (Constitution I). The Mac backend runs in-process and is
  not a sidecar service.
- Docker on macOS can't reach the Mac GPU, so Docker remains Linux/CUDA only.
- Command-line synthesis (`infer.py`) on a Mac is out of scope for this release (FR-005a). A
  later release may add it by reusing the server's Mac backend.
- Python 3.12 and uv stay the toolchain on every platform.
- The BreezeBlue Research and Non-Commercial License covers the weights in every format.
  Converting them, or using a community conversion, doesn't change that.
- Mac support ships as a minor version bump, since it adds a platform without changing the API.
  The changelog records it.
- The reference machine for every Mac measurement is an Apple M5 with 16 GB of memory.
