// Health live gate (tasks.md T006, T029): the provider loads against this run's server addresses.
// checkReady() (src/provider.js) does two things in sequence: it logs breeze.health once the HTTP
// health check succeeds, then calls refreshVoices() — which can itself fail even though the health
// check passed, in which case checkReady() logs breeze.check_failed anyway. So a pass here needs
// breeze.health followed by voices.refreshed, not breeze.health alone (review pass 3, finding 2).
// Also checks for no toast and no CORS console error. Matches quickstart.md Scenario 1.6.
//
// run.mjs navigates to SillyTavern once, before any phase, so it can snapshot settings before this
// phase's selectBreezeProvider() starts changing them (review pass 1, finding 7); this phase assumes
// the page is already loaded.
import { selectBreezeProvider, expectAnyEvent, normalizeUrl, readToasts, makeRecorder } from '../lib.mjs';

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

        // selectBreezeProvider waits for "TTS Provider Loaded" itself. Re-reading #tts_status
        // afterwards races SillyTavern replacing it with "Successfully applied settings".
        const since = await selectBreezeProvider(page, config, events);
        step('provider reports "TTS Provider Loaded"', true);

        // Only events from this run's own refresh-triggered checkReady() count: selecting Breeze and
        // enabling TTS both ran their own checkReady() first, against whatever URL a previous session
        // saved, which would otherwise get credited to this run (review pass 1, finding 8). Compare
        // with normalizeUrl since a saved address can differ only in whitespace or a trailing slash
        // from config.httpUrl (review pass 2, finding 8). A single synchronous read right after
        // selectBreezeProvider would be unreliable: if Breeze (and "TTS Provider Loaded") was already
        // active before this run, that wait resolves immediately, before this run's own
        // refresh-triggered checkReady() has necessarily logged anything yet — so this polls instead.
        const thisRun = (e) => normalizeUrl(e.httpUrl) === normalizeUrl(config.httpUrl);
        const healthEvent = await expectAnyEvent(
            step, events, since, ['breeze.health', 'breeze.check_failed'], thisRun, 20000,
            'breeze.health (not breeze.check_failed) logged for this run\'s httpUrl',
            (event) => event.event === 'breeze.health',
        );

        if (healthEvent?.event === 'breeze.health') {
            // checkReady() only reports success once refreshVoices() (called right after logging
            // breeze.health) has also succeeded; that step can fail on its own, in which case
            // checkReady() still logs breeze.check_failed despite the health check itself passing.
            const healthEventIndex = events.indexOf(healthEvent);
            await expectAnyEvent(
                step, events, healthEventIndex + 1, ['voices.refreshed', 'breeze.check_failed'], () => true, 20000,
                'checkReady() completes end to end (voices.refreshed, not breeze.check_failed)',
                (event) => event.event === 'voices.refreshed',
            );
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
