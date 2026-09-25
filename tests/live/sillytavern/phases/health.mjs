// Health live gate (tasks.md T006, T029): the provider loads against this run's server addresses,
// logs a `breeze.health` event and no `breeze.check_failed`, with no toast and no CORS console
// error. Matches quickstart.md Scenario 1.6.
import { openSillyTavern, selectBreezeProvider, readToasts, makeRecorder } from '../lib.mjs';

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @param {object[] & {errors: string[]}} events breeze console events, from captureBreezeEvents.
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function health(page, config, events) {
    const { results, step } = makeRecorder();

    // A thrown error (usually a wait timeout because no Breeze server is running) still leaves the
    // steps already recorded in `results` intact, and lets the next composed phase run instead of
    // aborting the whole `run.mjs` invocation.
    try {
        await openSillyTavern(page, config);
        step('SillyTavern loaded with the Breeze provider registered', true);

        await selectBreezeProvider(page, config);
        const status = await page.$eval('#tts_status', (e) => e.textContent);
        step('provider reports "TTS Provider Loaded"', status.includes('TTS Provider Loaded'), status.trim());

        const healthEvent = events.find((e) => e.event === 'breeze.health');
        step('breeze.health event logged', Boolean(healthEvent), JSON.stringify(healthEvent ?? null));

        const checkFailed = events.find((e) => e.event === 'breeze.check_failed');
        step('no breeze.check_failed event', !checkFailed, JSON.stringify(checkFailed ?? null));

        const toasts = await readToasts(page);
        step('no error toast after loading the provider', !toasts.hasError, toasts.text.slice(0, 120));

        const corsErrors = events.errors.filter((e) => /cors/i.test(e));
        step('no CORS errors in the page console', corsErrors.length === 0, corsErrors.slice(0, 3).join(' | '));
    } catch (error) {
        step('health phase aborted', false, error.message);
    }

    return results;
}
