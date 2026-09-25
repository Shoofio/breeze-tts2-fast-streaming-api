// Shared helpers for the SillyTavern live-test harness (tasks.md T005), adapted from
// <SillyTavern checkout>/specs/001-breeze-tts-provider/us*-browser-validation.mjs and
// make-validation-chat.mjs. Event names and selectors were cross-checked against the extension
// source at <SillyTavern checkout>/extensions/SillyTavern-BreezeTTS/src/*.js (see the T004-T010 report for
// the specific lines).
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
 * @param {import('playwright-core').Page} page
 * @returns {object[] & {errors: string[]}} the events array; `events.errors` holds console/page errors.
 */
export function captureBreezeEvents(page) {
    const events = [];
    events.errors = [];
    page.on('console', async (msg) => {
        if (msg.type() === 'error') events.errors.push(msg.text());
        // The extension always logs `console.debug('breeze', payload)`; the first arg is the literal
        // string 'breeze', the second is the JSON payload.
        if (msg.text().startsWith('breeze')) {
            try {
                const arg = msg.args()[1];
                if (arg) events.push(await arg.jsonValue());
            } catch {
                /* the page navigated away mid-read; drop this one event rather than fail the run */
            }
        }
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

/** Navigates to SillyTavern and waits for the Breeze provider's settings block to be in the DOM. */
export async function openSillyTavern(page, config) {
    await page.goto(config.stUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#breeze_settings', { state: 'attached', timeout: 90000 });
}

/**
 * Selects "Breeze" in the framework's TTS provider dropdown, enables TTS, points it at this run's
 * server addresses, and waits for the provider to report ready.
 */
export async function selectBreezeProvider(page, config) {
    await page.evaluate(() => { $('#tts_provider').val('Breeze').trigger('change'); });
    await page.waitForSelector('#breeze_settings', { state: 'attached', timeout: 15000 });
    await page.evaluate(() => { if (!$('#tts_enabled').prop('checked')) $('#tts_enabled').trigger('click'); });
    await page.evaluate(([httpUrl, wsUrl]) => {
        $('#breeze_http_url').val(httpUrl).trigger('input');
        $('#breeze_ws_url').val(wsUrl).trigger('input');
    }, [config.httpUrl, config.wsUrl]);
    await page.evaluate(() => { $('#tts_refresh').trigger('click'); });
    await page.waitForFunction(() => $('#tts_status').text().includes('TTS Provider Loaded'), null, { timeout: 30000 });
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
 * `specs/003-cpp-compatible-api/research/live-<name>.md`: a markdown summary table for people, plus
 * the raw results as a fenced JSON block so later phases (T011) can diff behavior mechanically.
 */
export function writeRecord(name, results) {
    const outDir = path.join(repoRoot, 'specs/003-cpp-compatible-api/research');
    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, `live-${name}.md`);

    const passed = results.filter((r) => r.ok).length;
    const rows = results
        .map((r) => `| ${r.ok ? 'ok' : 'FAIL'} | ${r.step} | ${r.detail.replace(/\|/g, '\\|')} |`)
        .join('\n');
    const body = `# Live SillyTavern run: ${name}

Generated ${new Date().toISOString()}. ${passed}/${results.length} steps passed.

| Result | Step | Detail |
| --- | --- | --- |
${rows}

\`\`\`json
${JSON.stringify(results, null, 2)}
\`\`\`
`;
    fs.writeFileSync(outPath, body);
    return outPath;
}

export { sleep };
