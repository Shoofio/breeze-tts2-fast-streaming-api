// Configuration for the SillyTavern live-test harness (specs/003-cpp-compatible-api/tasks.md T004).
//
// Every value here can be overridden by an environment variable, so the harness can point at a
// different SillyTavern instance, Breeze server or Playwright install without editing this file.
// Defaults were confirmed to exist on this machine on 2026-09-24 (see the T004 report):
// `~/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core` (playwright-core 1.63.0) and
// `~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`.
import os from 'node:os';
import path from 'node:path';

/** Expands a leading `~` the way a shell would; only the default values need it. */
function expandHome(value) {
    if (value.startsWith('~')) return path.join(os.homedir(), value.slice(1));
    return value;
}

const chromiumDir = expandHome(process.env.CHROMIUM_PATH ?? '~/.cache/ms-playwright/chromium-1234');

export const config = {
    // SillyTavern under test. Never point this at anything but the local dev container: the
    // safety rules in tasks.md standing rule 7 assume the "Breeze validation" chat under Seraphina.
    stUrl: process.env.ST_URL ?? 'http://127.0.0.1:8000/',

    // Breeze server endpoints (contracts/http-api.md, contracts/ws-api.md).
    httpUrl: process.env.BREEZE_HTTP_URL ?? 'http://127.0.0.1:8080',
    wsUrl: process.env.BREEZE_WS_URL ?? 'ws://127.0.0.1:8081',

    // Playwright is not a project dependency (R16: this harness borrows the npx-cached install the
    // SillyTavern validation scripts already use, rather than adding a devDependency).
    playwrightCorePath: expandHome(
        process.env.PLAYWRIGHT_CORE_PATH ?? '~/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core',
    ),
    // Directory form (as tasks.md T004 lists it); the executable lives at a fixed subpath inside it.
    chromiumPath: chromiumDir,
    chromiumExecutablePath: process.env.CHROMIUM_EXECUTABLE_PATH ?? path.join(chromiumDir, 'chrome-linux64', 'chrome'),

    headless: process.env.HEADLESS !== 'false',
    launchArgs: ['--autoplay-policy=no-user-gesture-required'],

    // Reference voice samples (WAV + transcript) used for voice registration and throwaway uploads.
    voicesDir: process.env.REFERENCE_VOICES_DIR ?? '$REFERENCE_VOICES_DIR',
};
