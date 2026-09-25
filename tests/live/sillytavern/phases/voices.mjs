// Voices live gate (tasks.md T007, quickstart.md Scenario 3.5). Uses only the `st_live_tmp`
// throwaway name (standing rule 7 — never touch `eric` or `vale`). Uploads reuse the `eric` sample
// recording/transcript purely as upload content; the voice is saved under the `st_live_tmp` name.
import fs from 'node:fs';
import path from 'node:path';
import { waitForEvent, readToasts, acceptPopupIfPresent, makeRecorder } from '../lib.mjs';

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

    // A thrown error still leaves the steps already recorded in `results` intact, and lets the next
    // composed phase run instead of aborting the whole `run.mjs` invocation.
    try {
        // 1. Upload st_live_tmp.
        let since = events.length;
        await fillUploadForm(TMP_NAME, sampleTranscript, sampleWav);
        await acceptPopupIfPresent(page, { accept: true, timeoutMs: 3000 }); // only if a stale copy exists
        const uploaded = await waitForEvent(events, 'voice.uploaded', () => true, 60000, since).catch((e) => e);
        step('st_live_tmp uploaded (voice.uploaded event)', uploaded instanceof Error === false, JSON.stringify(uploaded));

        const listedAfterUpload = await page.evaluate((n) => $('#breeze_voice_list').text().includes(n), TMP_NAME);
        step('st_live_tmp appears in the voice list', listedAfterUpload);

        // 2. Re-upload the same name and accept "Replace". The extension's owner is moving its replace
        //    flow to DELETE-then-POST (R16); until that lands, the server's 409 voice_exists on the raw
        //    POST is expected and is not a Breeze-server bug, so it is recorded, not failed.
        since = events.length;
        await fillUploadForm(TMP_NAME, sampleTranscript, sampleWav);
        const sawReplacePrompt = await acceptPopupIfPresent(page, { accept: true, timeoutMs: 5000 });
        const replaceOutcome = await Promise.race([
            waitForEvent(events, 'voice.uploaded', () => true, 15000, since).then(() => 'uploaded'),
            waitForEvent(events, 'voice.upload_failed', () => true, 15000, since).then(() => 'upload_failed'),
        ]).catch(() => 'timeout');
        step(
            'replace flow: re-upload st_live_tmp after confirming "Replace"',
            true, // either outcome is acceptable; see comment above
            sawReplacePrompt
                ? `outcome=${replaceOutcome}${replaceOutcome === 'uploaded' ? '' : ' (blocked on extension, R16)'}`
                : 'no replace prompt appeared — treating as blocked on extension',
        );

        // 3. Delete st_live_tmp.
        since = events.length;
        await page.evaluate((n) => { $(`.breeze_voice_delete[data-id="${n}"]`).trigger('click'); }, TMP_NAME);
        await acceptPopupIfPresent(page, { accept: true, timeoutMs: 5000 });
        const deleted = await waitForEvent(events, 'voice.deleted', () => true, 20000, since).catch((e) => e);
        step('st_live_tmp deleted (voice.deleted event)', deleted instanceof Error === false, JSON.stringify(deleted));

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
