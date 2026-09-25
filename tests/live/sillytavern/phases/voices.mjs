// Voices live gate (tasks.md T007, quickstart.md Scenario 3.5). Uses only the `st_live_tmp`
// throwaway name (standing rule 7 — never touch `eric` or `vale`). Uploads reuse the `eric` sample
// recording/transcript purely as upload content; the voice is saved under the `st_live_tmp` name.
import fs from 'node:fs';
import path from 'node:path';
import { expectEvent, readToasts, acceptPopupIfPresent, makeRecorder } from '../lib.mjs';

const TMP_NAME = 'st_live_tmp';

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @param {object[] & {errors: string[]}} events
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function voices(page, config, events) {
    const { results, step } = makeRecorder();

    const sampleWav = path.join(config.voicesDir, 'eric', 'eric.wav');
    const sampleTranscript = fs.readFileSync(path.join(config.voicesDir, 'eric', 'eric.txt'), 'utf8').trim();

    const fillUploadForm = async (name, transcript, file) => {
        if (file) await page.setInputFiles('#breeze_upload_file', file);
        await page.evaluate(([t, n]) => {
            $('#breeze_upload_transcript').val(t);
            $('#breeze_upload_name').val(n);
        }, [transcript, name]);
        await page.evaluate(() => { $('#breeze_upload_save').trigger('click'); });
    };

    // A thrown error (a page.evaluate failure, not an event wait — those are caught individually
    // below) still leaves the steps already recorded in `results` intact, and lets the next composed
    // phase run instead of aborting the whole `run.mjs` invocation.
    try {
        // 1. Upload st_live_tmp.
        let since = events.length;
        await fillUploadForm(TMP_NAME, sampleTranscript, sampleWav);
        await acceptPopupIfPresent(page, { accept: true, timeoutMs: 3000 }); // only if a stale copy exists
        await expectEvent(step, events, since, 'voice.uploaded', () => true, 60000, 'st_live_tmp uploaded (voice.uploaded event)');
        // refreshVoices() (and its `voices.refreshed` log) runs right after voice.uploaded, before the
        // extension re-renders #breeze_voice_list — read the list only once that has happened.
        await expectEvent(step, events, since, 'voices.refreshed', () => true, 15000, 'voice list refreshed after upload');

        const listedAfterUpload = await page.evaluate((n) => $('#breeze_voice_list').text().includes(n), TMP_NAME);
        step('st_live_tmp appears in the voice list', listedAfterUpload);

        // 2. Re-upload the same name and accept "Replace". The extension's DELETE-then-POST replace
        //    flow has landed (src/provider.js ~415-450: onUploadClick logs voice.replaced right after
        //    the old copy is deleted, then voice.uploaded once the new one is saved), so this must
        //    succeed now — a timeout or a voice.upload_failed is a real failure, not tolerated. 60 s
        //    timeouts (review pass 2, finding 10), matching the first upload's own generous timeout.
        since = events.length;
        await fillUploadForm(TMP_NAME, sampleTranscript, sampleWav);
        await acceptPopupIfPresent(page, { accept: true, timeoutMs: 5000 });
        await expectEvent(step, events, since, 'voice.replaced', () => true, 60000, 'st_live_tmp replaced (voice.replaced event)');
        await expectEvent(step, events, since, 'voice.uploaded', () => true, 60000, 'st_live_tmp re-uploaded after replace (voice.uploaded event)');
        const uploadFailed = events.slice(since).find((e) => e.event === 'voice.upload_failed');
        step('replace did not fail (no voice.upload_failed)', !uploadFailed, JSON.stringify(uploadFailed ?? null));
        await expectEvent(step, events, since, 'voices.refreshed', () => true, 60000, 'voice list refreshed after replace');

        // 3. Delete st_live_tmp.
        since = events.length;
        await page.evaluate((n) => { $(`.breeze_voice_delete[data-id="${n}"]`).trigger('click'); }, TMP_NAME);
        await acceptPopupIfPresent(page, { accept: true, timeoutMs: 5000 });
        await expectEvent(step, events, since, 'voice.deleted', () => true, 20000, 'st_live_tmp deleted (voice.deleted event)');
        await expectEvent(step, events, since, 'voices.refreshed', () => true, 15000, 'voice list refreshed after delete');

        const listedAfterDelete = await page.evaluate((n) => $('#breeze_voice_list').text().includes(n), TMP_NAME);
        step('st_live_tmp no longer in the voice list', !listedAfterDelete);

        const toasts = await readToasts(page);
        step('no error toast across the upload/replace/delete run', !toasts.hasError, toasts.text.slice(0, 160));

        // 4. Safety check: eric and vale were never touched.
        const listText = await page.$eval('#breeze_voice_list', (e) => e.textContent);
        step('eric is still listed', listText.includes('eric'), listText.trim().slice(0, 120));
        step('vale is still listed', listText.includes('vale'), listText.trim().slice(0, 120));
    } catch (error) {
        step('voices phase aborted', false, error.message);
    }

    return results;
}
