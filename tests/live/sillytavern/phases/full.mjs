// Full WebSocket-session live gate (tasks.md T009, quickstart.md Scenario 4.4), run inside the
// "Breeze validation" chat under Seraphina. Exercises narration, stop/cancel, voice preview, the
// server's queue signal, and cfg_scale settings end to end through the extension.
import { openValidationChat, waitForEvent, readToasts, sleep, makeRecorder } from '../lib.mjs';

const narrateLast = (page) => page.evaluate(() => { $('.mes').last().find('.mes_narrate').trigger('click'); });
const stop = (page) => page.evaluate(() => { $('#tts_media_control').trigger('click'); });
const setBaseline = (page, value) => page.evaluate((v) => { $('#breeze_guidance_baseline').val(v).trigger('input'); }, value);

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @param {object[] & {errors: string[]}} events
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function full(page, config, events) {
    const { results, step } = makeRecorder();

    // A thrown error still leaves the steps already recorded in `results` intact, and lets the next
    // composed phase run instead of aborting the whole `run.mjs` invocation.
    try {
        await openValidationChat(page);
        step('"Breeze validation" chat open under Seraphina', true);

        // 1. Narrate the last message (it already contains quoted dialogue, as the validation chat is
        //    written that way). Expect synth.request, then a synth.event of type 'started', then synth.done.
        await stop(page);
        await sleep(300);
        let since = events.length;
        await narrateLast(page);
        const request = await waitForEvent(events, 'synth.request', () => true, 20000, since);
        step('synth.request logged for the narration', Boolean(request), JSON.stringify(request));
        const started = await waitForEvent(events, 'synth.event', (e) => e.type === 'started', 15000, since);
        step('synth.event "started" logged', Boolean(started), JSON.stringify(started));
        const done = await waitForEvent(events, 'synth.done', () => true, 60000, since);
        const firstAudio = events.slice(since).find((e) => e.event === 'synth.first_audio');
        step('synth.done logged, audio delivered', done.bytes > 0, `bytes=${done.bytes} frames=${done.frames} firstAudioMs=${firstAudio?.ms}`);

        // 2. Narrate again, then stop mid-flight: expect synth.cancelled and no error toast.
        since = events.length;
        await narrateLast(page);
        await waitForEvent(events, 'synth.event', (e) => e.type === 'started', 15000, since);
        await stop(page);
        const cancelled = await waitForEvent(events, 'synth.cancelled', () => true, 20000, since);
        step('stop cancels the active session (synth.cancelled)', Boolean(cancelled), JSON.stringify(cancelled));
        const toastsAfterCancel = await readToasts(page);
        step('no error toast after a user-initiated cancel', !toastsAfterCancel.hasError, toastsAfterCancel.text.slice(0, 120));

        // 3. Voice preview for eric (never eric's saved recording — this only requests synthesis).
        since = events.length;
        await page.evaluate(() => { window.tts_preview('eric'); });
        const preview = await waitForEvent(events, 'synth.done', () => true, 60000, since);
        step('voice preview for eric synthesizes', preview.bytes > 0, `bytes=${preview.bytes}`);

        // 4. Two back-to-back narrations: the second should be reported queued while the first still
        //    holds the GPU (contracts/ws-api.md `queued`; src/breeze-ws.js case 'queued').
        await stop(page);
        await sleep(300);
        since = events.length;
        await narrateLast(page);
        await narrateLast(page);
        const queued = await waitForEvent(events, 'synth.event', (e) => e.type === 'queued', 20000, since).catch((e) => e);
        step('second back-to-back narration is queued', queued instanceof Error === false, JSON.stringify(queued));
        await waitForEvent(events, 'synth.done', () => true, 60000, since).catch(() => null); // let the pair settle before moving on

        // 5. cfg_scale via the baseline guidance setting (no direction or vocal-event tag active, so
        //    the request's cfgScale reflects the baseline directly — src/guidance.js pickGuidance()).
        for (const cfgScale of [1, 4, 7.5]) {
            await stop(page);
            await sleep(300);
            await setBaseline(page, cfgScale);
            since = events.length;
            await narrateLast(page);
            const req = await waitForEvent(events, 'synth.request', () => true, 20000, since);
            await waitForEvent(events, 'synth.done', () => true, 60000, since);
            step(`cfg_scale ${cfgScale} reaches the request`, req.cfgScale === cfgScale, `cfgScale=${req.cfgScale}`);
        }
    } catch (error) {
        step('full phase aborted', false, error.message);
    }

    return results;
}
