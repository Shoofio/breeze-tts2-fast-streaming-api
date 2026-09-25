// Health live gate (tasks.md T006, T029): the provider loads against this run's server addresses,
// logs a `breeze.health` event and no `breeze.check_failed`, with no toast and no CORS console
// error. Matches quickstart.md Scenario 1.6.
//
// run.mjs navigates to SillyTavern once, before any phase, so it can snapshot settings before this
// phase's selectBreezeProvider() starts changing them (review pass 1, finding 7); this phase assumes
// the page is already loaded.
import { selectBreezeProvider, waitForAnyEvent, normalizeUrl, readToasts, makeRecorder } from '../lib.mjs';

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @param {object[] & {errors: string[]}} events breeze console events, from captureBreezeEvents.
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function health(page, config, events) {
    const { results, step } = makeRecorder();

    try {
        const hasOption = await page.evaluate(() => $('#tts_provider option[value="Breeze"]').length > 0);
        step('SillyTavern loaded with the Breeze provider registered', hasOption);

        const since = await selectBreezeProvider(page, config, events);
        const status = await page.$eval('#tts_status', (e) => e.textContent);
        step('provider reports "TTS Provider Loaded"', status.includes('TTS Provider Loaded'), status.trim());

        // Only events from this run's own refresh-triggered checkReady() count: selecting Breeze and
        // enabling TTS both ran their own checkReady() first, against whatever URL a previous session
        // saved, which would otherwise get credited to this run (review pass 1, finding 8). Compare
        // with normalizeUrl since a saved address can differ only in whitespace or a trailing slash
        // from config.httpUrl (review pass 2, finding 8).
        const thisRun = (e) => normalizeUrl(e.httpUrl) === normalizeUrl(config.httpUrl);
        // A single synchronous read right after selectBreezeProvider is unreliable: if Breeze (and
        // "TTS Provider Loaded") was already active before this run, that wait resolves immediately,
        // before this run's own refresh-triggered checkReady() has necessarily logged anything yet.
        // Poll for either outcome instead, and only breeze.health counts as a pass (review pass 2,
        // finding 2).
        try {
            const outcome = await waitForAnyEvent(events, ['breeze.health', 'breeze.check_failed'], thisRun, 20000, since);
            step('breeze.health (not breeze.check_failed) logged for this run\'s httpUrl', outcome.event === 'breeze.health', JSON.stringify(outcome));
        } catch (error) {
            step('breeze.health (not breeze.check_failed) logged for this run\'s httpUrl', false, error.message);
        }

        const toasts = await readToasts(page);
        step('no error toast after loading the provider', !toasts.hasError, toasts.text.slice(0, 120));

        const corsErrors = events.errors.filter((e) => /cors/i.test(e));
        step('no CORS errors in the page console', corsErrors.length === 0, corsErrors.slice(0, 3).join(' | '));
    } catch (error) {
        step('health phase aborted', false, error.message);
    }

    return results;
}
