// Full WebSocket-session live gate (tasks.md T009, quickstart.md Scenario 4.4), run inside the
// "Breeze validation" chat under Seraphina. Exercises narration, stop/cancel, voice preview, the
// server's queue signal, and cfg_scale settings end to end through the extension.
import {
    openValidationChat, expectEvent, expectAnyEvent, readToasts, sleep, makeRecorder, holdGpuWithSession,
} from '../lib.mjs';

// A few hundred characters keep the server generating for several seconds (RTF ~0.3-0.4 per the
// baseline in research/), long enough for the page's own narration to arrive and get queued behind it.
const LONG_HOLD_TEXT = 'The lighthouse keeper climbed the spiral stairs before dawn, counting each '
    + 'worn stone step out of habit rather than need. Fog rolled in from the strait, thick enough to '
    + 'swallow the beam whole, so he lit the lamp early and settled in to wait for the first ships. '
    + 'Somewhere below, the sea kept its own patient rhythm against the rocks, indifferent to the hour.';

const narrateLast = (page) => page.evaluate(() => { $('.mes').last().find('.mes_narrate').trigger('click'); });
const setBaseline = (page, value) => page.evaluate((v) => { $('#breeze_guidance_baseline').val(v).trigger('input'); }, value);

/**
 * Stops playback only when it's actually running. `#tts_media_control`'s class mirrors
 * `!audioElement.paused || isTtsProcessing()` (SillyTavern's own updateUiAudioPlayState() in
 * public/scripts/extensions/tts/index.js): `fa-stop-circle` while active, `fa-circle-play` while
 * idle. Clicking it while idle doesn't no-op — onAudioControlClicked() instead *starts* narrating the
 * last message (review pass 1, finding 3), which is the opposite of what a "reset before the next
 * step" call means here.
 */
async function stop(page) {
    const active = await page.evaluate(() => $('#tts_media_control').hasClass('fa-stop-circle'));
    if (active) await page.evaluate(() => { $('#tts_media_control').trigger('click'); });
}

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @param {object[] & {errors: string[]}} events
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function full(page, config, events) {
    const { results, step } = makeRecorder();

    try {
        await openValidationChat(page);
        step('"Breeze validation" chat open under Seraphina', true);

        // 1. Narrate the last message (it already contains quoted dialogue, as the validation chat is
        //    written that way). Expect synth.request, then a synth.event of type 'started', then synth.done.
        await stop(page);
        await sleep(300);
        let since = events.length;
        await narrateLast(page);
        await expectEvent(step, events, since, 'synth.request', () => true, 20000, 'synth.request logged for the narration');
        await expectEvent(step, events, since, 'synth.event', (e) => e.type === 'started', 15000, 'synth.event "started" logged');
        const done = await expectEvent(step, events, since, 'synth.done', () => true, 60000, 'synth.done logged, audio delivered');
        if (done) step('audio bytes were delivered', done.bytes > 0, `bytes=${done.bytes} frames=${done.frames}`);

        // 2. Narrate again, then stop mid-flight. Either outcome is legitimate: a cancel, or the
        //    narration finishing on its own first (recorded explicitly, not mis-reported as a failed
        //    cancel — review pass 1, finding 4).
        since = events.length;
        await narrateLast(page);
        await expectEvent(step, events, since, 'synth.event', (e) => e.type === 'started', 15000, 'narration (for the cancel test) reaches "started"');
        await stop(page);
        await expectAnyEvent(
            step, events, since, ['synth.cancelled', 'synth.done'], 20000,
            'stop cancels the active session, or the narration finishes first',
        );
        const toastsAfterCancel = await readToasts(page);
        step('no error toast after the cancel attempt', !toastsAfterCancel.hasError, toastsAfterCancel.text.slice(0, 120));

        // 3. Voice preview for eric (this only requests synthesis; the saved recording is untouched).
        since = events.length;
        await page.evaluate(() => { window.tts_preview('eric'); });
        const preview = await expectEvent(step, events, since, 'synth.done', () => true, 60000, 'voice preview for eric synthesizes');
        if (preview) step('preview delivered audio bytes', preview.bytes > 0, `bytes=${preview.bytes}`);

        // 4. A concurrent session holds the GPU (contracts/ws-api.md `queued`; R14 GpuGate — one
        //    worker across every session, browser or not). A Node-side WebSocket client opens a real
        //    second session and keeps it generating, so the page's own narration has something to
        //    queue behind (review pass 1, finding 2 — two page-side narrations can't do this, since
        //    starting the second cancels the extension's own in-flight client session first).
        await stop(page);
        await sleep(300);
        const holder = holdGpuWithSession(config, { voiceId: 'vale', text: LONG_HOLD_TEXT });
        try {
            await holder.started;
            since = events.length;
            await narrateLast(page);
            await expectEvent(step, events, since, 'synth.event', (e) => e.type === 'queued', 20000, 'narration is queued while the Node client holds the GPU');
        } finally {
            holder.cancel();
            holder.close();
        }
        // Now that the GPU is free, the page's own narration should complete normally.
        await expectEvent(step, events, since, 'synth.done', () => true, 60000, 'queued narration completes once the GPU frees up');

        // 5. cfg_scale via the baseline guidance setting. Direction and vocal-event tags are pinned
        //    off and delivery is pinned to "buffer" (only that mode's synth.first_audio carries `ms`;
        //    see bufferedTts() in src/provider.js) so the request's cfgScale reflects the baseline
        //    directly (src/guidance.js pickGuidance()) — run.mjs restores all of this afterward
        //    (review pass 1, findings 7 and 9). The first value (4) differs from the extension's
        //    default baseline (1), so a setBaseline that silently no-ops can't pass by coincidence.
        await page.evaluate(() => {
            $('#breeze_direction_enabled').prop('checked', false).trigger('change');
            $('#breeze_vocal_events_enabled').prop('checked', false).trigger('change');
            $('#breeze_delivery_mode').val('buffer').trigger('change');
        });
        for (const cfgScale of [4, 7.5, 1]) {
            await stop(page);
            await sleep(300);
            await setBaseline(page, cfgScale);
            since = events.length;
            await narrateLast(page);
            const req = await expectEvent(step, events, since, 'synth.request', () => true, 20000, `cfg_scale ${cfgScale}: synth.request logged`);
            await expectEvent(step, events, since, 'synth.done', () => true, 60000, `cfg_scale ${cfgScale}: synth.done logged`);
            step(`cfg_scale ${cfgScale} reached the request`, req?.cfgScale === cfgScale, `cfgScale=${req?.cfgScale}`);
        }
    } catch (error) {
        step('full phase aborted', false, error.message);
    }

    return results;
}
