# Changelog

All notable changes to this project are documented here. See
`specs/003-cpp-compatible-api/spec.md` for the full breaking-changes list and rationale (IDs
`BC-nn`).

## Unreleased — 2.0.0

The previous Python HTTP API (the pre-2.0 request/response shapes) is removed. The server now
exposes a C++-server-compatible HTTP and WebSocket API; see the README and
`specs/003-cpp-compatible-api/contracts/http-api.md` for the full contract.

### Breaking changes

- **BC-18**: Every error response now uses the JSON envelope `{"error", "code"}`. A method the
  route doesn't support (including `OPTIONS` without CORS enabled) now returns `405` with an
  `Allow` header, instead of a bare `404`.
- **BC-19**: CORS allowlist entries are trimmed of surrounding whitespace before matching, so a
  spaced list (`"a, b"`) now matches `b`.
- **BC-20**: In CORS allowlist mode, every response carries `Vary: Origin`, not only responses
  that matched an allowed origin.
- **BC-21**: A CORS preflight now returns `204` only for a route that actually exists, advertising
  only that route's methods; a preflight for an unknown path gets `404`.
- **BC-22**: Mixing `*` into a CORS allowlist is a startup error instead of a silently dead entry.
- **BC-23**: A cross-origin browser `POST`/`DELETE` from a disallowed (or, with CORS off, any)
  origin now gets `403` before any work is done, instead of running and possibly writing a voice
  file.
- **BC-24**: `GET /health` reports `ws_port: 0` when the WebSocket failed to bind, instead of
  reporting a port that isn't actually listening.

Later phases append their own `BC-nn` entries to this section as they land.
