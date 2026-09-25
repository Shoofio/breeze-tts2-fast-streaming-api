// Shared helpers for the SillyTavern live-test harness (tasks.md T005), adapted from
// <SillyTavern checkout>/specs/001-breeze-tts-provider/us*-browser-validation.mjs and
// make-validation-chat.mjs. Event names and selectors were cross-checked against the extension
// source at <SillyTavern checkout>/extensions/SillyTavern-BreezeTTS/src/*.js and, for framework-level
// behavior (the media control button, checkReady's trigger), the running container's own
// public/scripts/extensions/tts/index.js (see the review-pass-1 fixup report for the specific lines).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Starts a headless Chromium instance from the npx-cached playwright-core install (R16 — this
 * harness has no Playwright devDependency of its own).
 *
 * @param {import('./config.mjs').config} config
 * @returns {Promise<{browser: import('playwright-core').Browser, page: import('playwright-core').Page}>}
 */
export async function launchBrowser(config) {
    const require = createRequire(path.join(config.playwrightCorePath, 'package.json'));
    const { chromium } = require('playwright-core');
    const browser = await chromium.launch({
        headless: config.headless,
        executablePath: config.chromiumExecutablePath,
        args: config.launchArgs,
    });
    const page = await browser.newPage();
    return { browser, page };
}

/**
 * Attaches console/pageerror listeners and returns the running list of parsed `breeze` events (see
 * src/log.js: `console.debug('breeze', {event, at, ...fields})`). Plain console/page errors are
 * kept on the same array as `.errors`, so a single object travels through every phase without
 * changing the `(page, config, events)` signature tasks.md T006-T009 describe.
 *
 * Every console message reserves its array slot synchronously, in the order Chromium fired it, then
 * fills the slot in once the JSHandles resolve. Those reads are async (a round trip to the page), so
 * without a synchronously-reserved slot two closely-spaced logs can be recorded out of order —
 * which would break `since = events.length` markers taken between actions. A slot that turns out not
 * to be a breeze event (wrong tag, or the page navigated away mid-read) is left without an `.event`
 * field, so it's inert: every consumer here matches on `.event`, never on array position.
 *
 * @param {import('playwright-core').Page} page
 * @returns {object[] & {errors: string[]}} the events array; `events.errors` holds console/page errors.
 */
export function captureBreezeEvents(page) {
    const events = [];
    events.errors = [];
    page.on('console', (msg) => {
        if (msg.type() === 'error') events.errors.push(msg.text());
        // Cheap text pre-filter, before touching `msg.args()` at all: the vast majority of console
        // messages are SillyTavern's own and have nothing to do with Breeze, so this skips creating
        // JSHandles (and reserving a slot) for them entirely (review pass 2, finding 9). The resolved
        // value of args[0] below is still the authoritative check — another message's rendered text
        // could coincidentally start the same way (review pass 1, finding 10).
        if (!msg.text().startsWith('breeze')) return;
        const args = msg.args();
        if (args.length < 2) {
            Promise.all(args.map((arg) => arg.dispose().catch(() => {})));
            return; // not a `console.debug('breeze', payload)` call after all
        }
        const slot = {};
        events.push(slot);
        (async () => {
            try {
                const tag = await args[0].jsonValue();
                if (tag !== 'breeze') return;
                Object.assign(slot, await args[1].jsonValue());
            } catch {
                /* the page navigated away mid-read; leave the slot empty rather than fail the run */
            } finally {
                await Promise.all(args.map((arg) => arg.dispose().catch(() => {})));
            }
        })();
    });
    page.on('pageerror', (err) => events.errors.push(`pageerror: ${err.message}`));
    return events;
}

/** Polls `events` until one matching `name` (and optional `predicate`) appears at or after `since`. */
export async function waitForEvent(events, name, predicate = () => true, timeoutMs = 45000, since = 0) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        const hit = events.slice(since).find((e) => e.event === name && predicate(e));
        if (hit) return hit;
        await sleep(200);
    }
    throw new Error(`timed out waiting for event ${name}`);
}

/**
 * Waits for an event and records the outcome as one step, rather than throwing and losing every
 * later step in the phase to an uncaught timeout (review pass 1, finding 4).
 *
 * @returns {Promise<object|null>} the event, or null when it never arrived (already recorded as a
 *   failed step; callers that need a field from it should optional-chain, e.g. `event?.cfgScale`).
 */
export async function expectEvent(step, events, since, name, predicate, timeoutMs, label) {
    try {
        const event = await waitForEvent(events, name, predicate, timeoutMs, since);
        step(label, true, JSON.stringify(event));
        return event;
    } catch (error) {
        step(label, false, error.message);
        return null;
    }
}

/**
 * Like {@link waitForEvent}, but resolves on the first of several event names — for places where
 * more than one outcome is legitimate (e.g. a narration finishing on its own before a stop click
 * reaches it).
 */
export async function waitForAnyEvent(events, names, predicate = () => true, timeoutMs = 45000, since = 0) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        const hit = events.slice(since).find((e) => names.includes(e.event) && predicate(e));
        if (hit) return hit;
        await sleep(200);
    }
    throw new Error(`timed out waiting for any of ${names.join(', ')}`);
}

/** {@link expectEvent}'s counterpart for {@link waitForAnyEvent}. */
export async function expectAnyEvent(step, events, since, names, predicate, timeoutMs, label) {
    try {
        const event = await waitForAnyEvent(events, names, predicate, timeoutMs, since);
        step(label, true, JSON.stringify(event));
        return event;
    } catch (error) {
        step(label, false, error.message);
        return null;
    }
}

/** Mirrors the extension's own normalizeUrl (src/provider.js): trims whitespace and strips a
 * trailing slash, so comparing a saved address to this run's config isn't defeated by formatting
 * (review pass 2, finding 8). */
export function normalizeUrl(value) {
    return String(value ?? '').trim().replace(/\/+$/, '');
}

/** Reads the toast tray. `hasError` checks toastr's own `toast-error` class rather than text, so it
 * doesn't depend on message wording. */
export async function readToasts(page) {
    return page.evaluate(() => {
        const container = document.getElementById('toast-container');
        return {
            text: container ? container.textContent.trim() : '',
            hasError: Boolean(container && container.querySelector('.toast-error')),
        };
    });
}

/**
 * Navigates to SillyTavern and waits for the framework's TTS provider dropdown to list "Breeze".
 * `#breeze_settings` is the wrong thing to wait for here (review pass 1, finding 1): the extension
 * only renders it once Breeze is the *selected* provider, which may not be true yet on a fresh page
 * load — the dropdown option is what tasks.md T005 actually specifies, and it exists regardless of
 * which provider is currently active.
 */
export async function openSillyTavern(page, config) {
    await page.goto(config.stUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#tts_provider option[value="Breeze"]', { state: 'attached', timeout: 90000 });
}

/**
 * Selects "Breeze" in the framework's TTS provider dropdown, enables TTS, points it at this run's
 * server addresses, and waits for the provider to report ready.
 *
 * @returns {Promise<number>} the `events` index at the moment the refresh click was sent, i.e. the
 *   first index that can possibly hold a `breeze.health`/`breeze.check_failed` for *this* run's
 *   addresses — selecting "Breeze" and enabling TTS both run their own `checkReady()` first, against
 *   whatever URL was saved from a previous session, before this function has set this run's URLs.
 */
export async function selectBreezeProvider(page, config, events) {
    await page.evaluate(() => { $('#tts_provider').val('Breeze').trigger('change'); });
    await page.waitForSelector('#breeze_settings', { state: 'attached', timeout: 15000 });
    await page.evaluate(() => { if (!$('#tts_enabled').prop('checked')) $('#tts_enabled').trigger('click'); });
    await page.evaluate(([httpUrl, wsUrl]) => {
        $('#breeze_http_url').val(httpUrl).trigger('input');
        $('#breeze_ws_url').val(wsUrl).trigger('input');
    }, [config.httpUrl, config.wsUrl]);
    const since = events.length;
    // #tts_refresh's handler calls the provider's onRefreshClick() (voices.refreshed) and then
    // initVoiceMap(), which runs checkReady() (breeze.health / breeze.check_failed) against the
    // URLs just set above (public/scripts/extensions/tts/index.js onRefreshClick, initVoiceMapInternal).
    await page.evaluate(() => { $('#tts_refresh').trigger('click'); });
    await page.waitForFunction(() => $('#tts_status').text().includes('TTS Provider Loaded'), null, { timeout: 30000 });
    return since;
}

/**
 * Opens the "Breeze validation" chat under Seraphina (tasks.md standing rule 7 — never any other
 * chat). Assumes SillyTavern has already loaded at least one chat, which happens a few seconds
 * after the initial page load.
 */
export async function openValidationChat(page) {
    await page.waitForFunction(() => SillyTavern.getContext().chat.length > 0, null, { timeout: 60000 });
    await page.evaluate(async () => {
        const ctx = SillyTavern.getContext();
        const index = ctx.characters.findIndex((c) => c.name === 'Seraphina');
        if (index < 0) throw new Error('Seraphina character not found');
        await ctx.selectCharacterById(index);
        await ctx.openCharacterChat('Breeze validation');
    });
    await page.waitForFunction(
        () => { const c = SillyTavern.getContext(); return c.getCurrentChatId() === 'Breeze validation' && c.chat.length > 0; },
        null,
        { timeout: 30000 },
    );
}

/** Accepts (or cancels) SillyTavern's own popup confirm dialog, if one appears within `timeoutMs`. */
export async function acceptPopupIfPresent(page, { accept = true, timeoutMs = 5000 } = {}) {
    const appeared = await page
        .waitForSelector('dialog[open]', { state: 'visible', timeout: timeoutMs })
        .then(() => true)
        .catch(() => false);
    if (appeared) await page.click(`dialog[open] .popup-button-${accept ? 'ok' : 'cancel'}`);
    return appeared;
}

/**
 * Snapshots the whole `tts` settings tree (`SillyTavern.getContext().extensionSettings.tts`), plus
 * the selected provider and the enabled flag, so `run.mjs` can put everything back afterward. This
 * replaces reading individual `#breeze_*` DOM fields (review pass 1, finding 7): those elements only
 * exist once Breeze has actually been selected once (see openSillyTavern above), so a DOM-based
 * snapshot missed everything whenever Breeze wasn't already the active provider (review pass 2,
 * finding 3) — cloning the underlying settings object needs no such precondition, and covers every
 * field the harness might touch (guidance, direction/vocal-event/inline-tag toggles, delivery mode,
 * per-voice style, ...) without having to name each one.
 */
export async function captureSettings(page) {
    return page.evaluate(() => {
        const ctx = SillyTavern.getContext();
        return {
            provider: $('#tts_provider').val(),
            ttsEnabled: $('#tts_enabled').prop('checked'),
            tts: structuredClone(ctx.extensionSettings.tts),
        };
    });
}

/**
 * Restores a snapshot from {@link captureSettings}. Writes the whole `tts` object back (undoing
 * anything the harness changed or newly created), saves it for real, then reselects the provider so
 * the live UI and the framework's in-memory provider instance match again.
 *
 * The explicit save matters (review pass 2, finding 1): SillyTavern's own `saveSettingsDebounced`
 * waits 1000 ms before writing to disk, which `browser.close()` would beat every time, so the restore
 * would silently never reach disk. `getContext()` only exposes the debounced form
 * (public/scripts/st-context.js), so this imports the real, awaitable `saveSettings()` directly from
 * `/script.js` (the container's public/script.js `export async function saveSettings(...)`, which
 * does a real `await fetch('/api/settings/save', ...)`) — the same "import the page's own module"
 * trick make-validation-chat.mjs already uses for `doNewChat()`.
 */
export async function restoreSettings(page, snapshot) {
    if (!snapshot) return;
    await page.evaluate(async (s) => {
        const ctx = SillyTavern.getContext();
        ctx.extensionSettings.tts = s.tts;
        const scriptModule = await import('/script.js');
        await scriptModule.saveSettings();
        // Reselect so the live UI and the framework's in-memory provider instance (closures inside
        // public/scripts/extensions/tts/index.js, not part of extensionSettings) match what was just
        // restored — loadTtsProvider() re-reads extension_settings.tts[name], already restored above.
        $('#tts_provider').val(s.provider).trigger('change');
        if ($('#tts_enabled').prop('checked') !== s.ttsEnabled) $('#tts_enabled').trigger('click');
    }, snapshot);
}

/**
 * A minimal Node-side client against Breeze's own WebSocket protocol (contracts/ws-api.md), used to
 * hold the server's single GPU slot busy from *outside* the browser page (review pass 1, finding 2):
 * two page-side narrations can't demonstrate `queued`, because starting the second one cancels the
 * extension's own in-flight client session before the server ever sees a competing request. Node 22+
 * has a global `WebSocket`; this repo's harness targets that (see the T004-T010 report).
 *
 * `started` resolves on the server's `speaking` message, not `started` (confusingly similarly named):
 * `started` only confirms the `start` message was accepted, before `end` has even been sent, so the
 * GPU may not be generating anything yet at that point — `speaking` is contracts/ws-api.md's signal
 * that a piece has actually begun (review pass 2, finding 6). It also has its own timeout, since a
 * caller awaiting it needs to move on (and record the failure) rather than hang forever.
 *
 * @param {import('./config.mjs').config} config
 * @param {{voiceId: string, text: string, cfgScale?: number, readyTimeoutMs?: number}} opts
 * @returns {{started: Promise<void>, cancel: () => void, close: () => void}}
 */
export function holdGpuWithSession(config, { voiceId, text, cfgScale = 1, readyTimeoutMs = 20000 }) {
    const socket = new WebSocket(config.wsUrl);
    let resolveStarted;
    let rejectStarted;
    const started = new Promise((resolve, reject) => { resolveStarted = resolve; rejectStarted = reject; });
    const readyTimer = setTimeout(() => {
        rejectStarted(new Error(`timed out waiting for the Node WS client to start speaking (${readyTimeoutMs} ms)`));
    }, readyTimeoutMs);
    const settle = (fn) => { clearTimeout(readyTimer); fn(); };

    socket.addEventListener('message', (event) => {
        if (typeof event.data !== 'string') return; // binary PCM frame; not needed here
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === 'ready') {
            socket.send(JSON.stringify({ type: 'start', voice_id: voiceId, cfg_scale: cfgScale }));
        } else if (message.type === 'started') {
            socket.send(JSON.stringify({ type: 'end', text }));
        } else if (message.type === 'speaking') {
            settle(resolveStarted);
        } else if (message.type === 'error') {
            settle(() => rejectStarted(new Error(`Breeze WS error: ${message.message ?? message.code}`)));
        }
    });
    socket.addEventListener('close', (event) => {
        if (event.code !== 1000) settle(() => rejectStarted(new Error(`Breeze WS closed unexpectedly (code ${event.code})`)));
    });

    return {
        started,
        cancel: () => { try { socket.send(JSON.stringify({ type: 'cancel' })); } catch { /* already closed */ } },
        close: () => { try { socket.close(); } catch { /* already closed */ } },
    };
}

/** Pushes one `{step, ok, detail}` result and echoes it to the console as each phase runs. */
export function makeRecorder() {
    const results = [];
    const step = (name, ok, detail = '') => {
        results.push({ step: name, ok, detail });
        console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`);
    };
    return { results, step };
}

/**
 * Writes one human-and-machine-readable record to
 * `specs/003-cpp-compatible-api/research/live-<name>.md`: a markdown summary table for people, the
 * raw results as a fenced JSON block so later phases (T011) can diff behavior mechanically, and,
 * when given, the settings snapshot `captureSettings` took — so if the automatic restore ever fails,
 * the exact pre-run `tts` settings tree is still on hand to repair by hand (review pass 2, "also give
 * a way to print the snapshot").
 */
export function writeRecord(name, results, snapshot = null) {
    const outDir = path.join(repoRoot, 'specs/003-cpp-compatible-api/research');
    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, `live-${name}.md`);

    const passed = results.filter((r) => r.ok).length;
    const rows = results
        .map((r) => `| ${r.ok ? 'ok' : 'FAIL'} | ${r.step} | ${r.detail.replace(/\|/g, '\\|')} |`)
        .join('\n');
    const snapshotSection = snapshot
        ? `\n## Settings snapshot (captured before this run; restore by hand if needed)\n\n\`\`\`json\n${JSON.stringify(snapshot, null, 2)}\n\`\`\`\n`
        : '';
    const body = `# Live SillyTavern run: ${name}

Generated ${new Date().toISOString()}. ${passed}/${results.length} steps passed.

| Result | Step | Detail |
| --- | --- | --- |
${rows}
${snapshotSection}
\`\`\`json
${JSON.stringify(results, null, 2)}
\`\`\`
`;
    fs.writeFileSync(outPath, body);
    return outPath;
}

export { sleep };
