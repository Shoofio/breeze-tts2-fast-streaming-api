// Full WebSocket-session live gate (tasks.md T009, quickstart.md Scenario 4.4), run inside the
// "Breeze validation" chat under Seraphina. Exercises narration, stop/cancel, voice preview, the
// server's queue signal, and cfg_scale settings end to end through the extension.
import {
    openValidationChat, expectEvent, waitForAnyEvent, readToasts, sleep, makeRecorder, holdGpuWithSession,
} from '../lib.mjs';

// A few hundred characters keep the server generating for several seconds (RTF ~0.3-0.4 per the
// baseline in research/), long enough for the page's own narration to arrive and get queued behind
// it, and long enough that the cancel test (below) has time to send its stop click before the
// narration finishes on its own.
const LONG_HOLD_TEXT = 'The lighthouse keeper climbed the spiral stairs before dawn, counting each '
    + 'worn stone step out of habit rather than need. Fog rolled in from the strait, thick enough to '
    + 'swallow the beam whole, so he lit the lamp early and settled in to wait for the first ships. '
    + 'Somewhere below, the sea kept its own patient rhythm against the rocks, indifferent to the hour.';

// A long narration runs at about real time on the C++ Q4 model (53 s of audio took 54 s), so a
// 60 s wait was too tight. Three minutes still catches a hang.
const SYNTH_DONE_TIMEOUT_MS = 180000;
const narrateLast = (page) => page.evaluate(() => { $('.mes').last().find('.mes_narrate').trigger('click'); });
const setBaseline = (page, value) => page.evaluate((v) => { $('#breeze_guidance_baseline').val(v).trigger('input'); }, value);

/**
 * Stops playback only when it's actually running (a single, immediate check) — used to reset state
 * before starting the next step, once earlier waits have already given the framework time to settle.
 * `#tts_media_control`'s class mirrors `!audioElement.paused || isTtsProcessing()` (SillyTavern's own
 * updateUiAudioPlayState() in public/scripts/extensions/tts/index.js): `fa-stop-circle` while active,
 * `fa-circle-play` while idle. Clicking it while idle doesn't no-op — onAudioControlClicked() instead
 * *starts* narrating the last message (review pass 1, finding 3), which is the opposite of what a
 * "reset before the next step" call means here.
 */
async function stop(page) {
    const active = await page.evaluate(() => $('#tts_media_control').hasClass('fa-stop-circle'));
    if (active) await page.evaluate(() => { $('#tts_media_control').trigger('click'); });
}

/**
 * Used only by the cancel test: polls up to `timeoutMs` for the stop icon before deciding there's
 * nothing to stop, since the framework's own job-processing state can lag slightly behind our WS
 * session's 'started' event — a single immediate check (like {@link stop}) risks finding it still
 * idle and sending no click at all. Returns whether a click was actually sent, so the caller can tell
 * a genuine cancel from a narration that had already finished before the click could land (review
 * pass 2, finding 5).
 */
async function stopIfPlaying(page, timeoutMs = 3000) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
        if (await page.evaluate(() => $('#tts_media_control').hasClass('fa-stop-circle'))) {
            await page.evaluate(() => { $('#tts_media_control').trigger('click'); });
            return true;
        }
        await sleep(150);
    }
    return false;
}

/** Adds one throwaway line as Seraphina (as us4-browser-validation.mjs does with /sendas), waiting
 * for it to land, then stops any auto-narration the framework may have started for it on its own. */
async function addThrowawayLine(page, text) {
    const before = await page.evaluate(() => SillyTavern.getContext().chat.length);
    await page.evaluate((t) => SillyTavern.getContext().executeSlashCommandsWithOptions(`/sendas name=Seraphina ${t}`), text);
    await page.waitForFunction((n) => SillyTavern.getContext().chat.length > n, before, { timeout: 10000 });
    await sleep(1500); // let the TTS extension's own auto-narration (if any) start, so stop() can catch it
    await stop(page);
}

/** Removes {@link addThrowawayLine}'s line, but only while it is still the last message, so a real
 * line in the "Breeze validation" chat can never be deleted (standing rule 7). */
const removeLastLine = (page, text) => page.evaluate(async (t) => {
    const { chat, executeSlashCommandsWithOptions } = SillyTavern.getContext();
    if (chat.length && chat[chat.length - 1].mes.trim() === t.trim()) {
        await executeSlashCommandsWithOptions('/del 1');
    }
}, text);

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
        const done = await expectEvent(step, events, since, 'synth.done', () => true, SYNTH_DONE_TIMEOUT_MS, 'synth.done logged, audio delivered');
        if (done) step('audio bytes were delivered', done.bytes > 0, `bytes=${done.bytes} frames=${done.frames}`);

        // 2. Narrate a long throwaway message, then stop mid-flight. A long message makes this
        //    reliable: the validation chat's own last message might be short enough to finish before
        //    the stop click lands (review pass 2, finding 5). If no click was sent, that's a failed
        //    cancel test, not a skipped pass; if a click was sent but only synth.done arrived (not
        //    synth.cancelled), that's a genuine failure too.
        try {
            await addThrowawayLine(page, LONG_HOLD_TEXT);
            since = events.length;
            await narrateLast(page);
            await expectEvent(step, events, since, 'synth.event', (e) => e.type === 'started', 15000, 'narration (for the cancel test) reaches "started"');
            const clickSent = await stopIfPlaying(page, 3000);
            step('stop click was sent (narration was still active)', clickSent);
            if (clickSent) {
                try {
                    const outcome = await waitForAnyEvent(events, ['synth.cancelled', 'synth.done'], () => true, 20000, since);
                    step('stop cancelled the active session (not synth.done)', outcome.event === 'synth.cancelled', JSON.stringify(outcome));
                } catch (error) {
                    step('stop cancelled the active session (not synth.done)', false, error.message);
                }
            } else {
                step(
                    'cancel test: nothing to cancel (the narration had already finished before the stop click could be sent)',
                    false,
                    'the throwaway message may need to be even longer, or the stop poll longer',
                );
            }
            const toastsAfterCancel = await readToasts(page);
            step('no error toast after the cancel attempt', !toastsAfterCancel.hasError, toastsAfterCancel.text.slice(0, 120));
        } finally {
            await removeLastLine(page, LONG_HOLD_TEXT).catch(() => {});
        }

        // 3. Voice preview for eric (this only requests synthesis; the saved recording is untouched).
        since = events.length;
        await page.evaluate(() => { window.tts_preview('eric'); });
        const preview = await expectEvent(step, events, since, 'synth.done', () => true, SYNTH_DONE_TIMEOUT_MS, 'voice preview for eric synthesizes');
        if (preview) step('preview delivered audio bytes', preview.bytes > 0, `bytes=${preview.bytes}`);

        // 4. A concurrent session holds the GPU (contracts/ws-api.md `queued`; R14 GpuGate — one
        //    worker across every session, browser or not). A Node-side WebSocket client opens a real
        //    second session and keeps it generating, so the page's own narration has something to
        //    queue behind (review pass 1, finding 2 — two page-side narrations can't do this, since
        //    starting the second cancels the extension's own in-flight client session first).
        await stop(page);
        await sleep(300);
        const holder = holdGpuWithSession(config, { voiceId: 'vale', text: LONG_HOLD_TEXT });
        let holderBusy = false;
        try {
            await holder.started; // resolves once the server reports 'speaking' (see lib.mjs)
            holderBusy = true;
            step('Node WS client is generating (holds the GPU)', true);
        } catch (error) {
            // Recorded, not thrown (review pass 2, finding 6): the queued check below is skipped, but
            // execution continues into the cfg_scale checks either way.
            step('Node WS client is generating (holds the GPU)', false, error.message);
        }
        if (holderBusy) {
            since = events.length;
            await narrateLast(page);
            await expectEvent(step, events, since, 'synth.event', (e) => e.type === 'queued', 20000, 'narration is queued while the Node client holds the GPU');
        }
        holder.cancel();
        holder.close();
        if (holderBusy) {
            // Now that the GPU is free, the page's own narration should complete normally.
            await expectEvent(step, events, since, 'synth.done', () => true, SYNTH_DONE_TIMEOUT_MS, 'queued narration completes once the GPU frees up');
        } else {
            step('queued narration completes once the GPU frees up', false, 'skipped: the Node client never confirmed it was generating');
        }

        // 5. cfg_scale via the baseline guidance setting. Direction, vocal-event tags and inline tags
        //    are pinned off, delivery is pinned to "buffer" (only that mode's synth.first_audio
        //    carries `ms`; see bufferedTts() in src/provider.js), and every known voice's style is
        //    cleared (buildRequest() folds a non-empty style into the instruction regardless of
        //    direction/tags — src/provider.js buildRequest(), src/guidance.js pickGuidance() — review
        //    pass 2, finding 7) so the request's cfgScale reflects the baseline directly — run.mjs
        //    restores all of this afterward (review pass 1, findings 7 and 9; review pass 2, finding
        //    1/3 for how). The first value (4) differs from the extension's default baseline (1), so a
        //    setBaseline that silently no-ops can't pass by coincidence.
        await page.evaluate(() => {
            $('#breeze_direction_enabled').prop('checked', false).trigger('change');
            $('#breeze_vocal_events_enabled').prop('checked', false).trigger('change');
            $('#breeze_inline_tags_enabled').prop('checked', false).trigger('change');
            $('#breeze_delivery_mode').val('buffer').trigger('change');
            const voices = SillyTavern.getContext().extensionSettings.tts.Breeze?.voices ?? {};
            for (const voiceId of Object.keys(voices)) voices[voiceId].style = '';
        });
        for (const cfgScale of [4, 7.5, 1]) {
            await stop(page);
            await sleep(300);
            await setBaseline(page, cfgScale);
            since = events.length;
            await narrateLast(page);
            const req = await expectEvent(step, events, since, 'synth.request', () => true, 20000, `cfg_scale ${cfgScale}: synth.request logged`);
            await expectEvent(step, events, since, 'synth.done', () => true, SYNTH_DONE_TIMEOUT_MS, `cfg_scale ${cfgScale}: synth.done logged`);
            step(`cfg_scale ${cfgScale}: no direction/tag active (hasInstruction false)`, req?.hasInstruction === false, `hasInstruction=${req?.hasInstruction}`);
            step(`cfg_scale ${cfgScale} reached the request`, req?.cfgScale === cfgScale, `cfgScale=${req?.cfgScale}`);
        }
    } catch (error) {
        step('full phase aborted', false, error.message);
    }

    return results;
}
