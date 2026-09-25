// Speech-over-HTTP live gate (tasks.md T008, quickstart.md Scenario 2.6): calls
// POST /v1/audio/speech from inside the SillyTavern page, so the request carries the page's own
// Origin and exercises the real CORS expose-headers path rather than a Node-side HTTP client.
import { makeRecorder } from '../lib.mjs';

/**
 * @param {import('playwright-core').Page} page
 * @param {import('../config.mjs').config} config
 * @returns {Promise<{step: string, ok: boolean, detail: string}[]>}
 */
export default async function speech(page, config) {
    const { results, step } = makeRecorder();

    // A thrown error still leaves the steps already recorded in `results` intact, and lets the next
    // composed phase run instead of aborting the whole `run.mjs` invocation.
    try {
        const ok = await page.evaluate(async (httpUrl) => {
            const body = new FormData();
            body.append('text', 'Hello there.');
            const res = await fetch(`${httpUrl}/v1/audio/speech`, { method: 'POST', body });
            const buf = await res.arrayBuffer();
            return {
                status: res.status,
                sampleRate: res.headers.get('X-Sample-Rate'),
                byteLength: buf.byteLength,
            };
        }, config.httpUrl);
        step('POST /v1/audio/speech returns 200', ok.status === 200, `status=${ok.status}`);
        step('X-Sample-Rate header is readable (expose-headers work)', Boolean(ok.sampleRate), `X-Sample-Rate=${ok.sampleRate}`);
        step('body is non-empty with an even byte length', ok.byteLength > 0 && ok.byteLength % 2 === 0, `bytes=${ok.byteLength}`);

        const bad = await page.evaluate(async (httpUrl) => {
            const body = new FormData();
            body.append('text', 'Hello there.');
            body.append('cfg_scale', 'banana');
            const res = await fetch(`${httpUrl}/v1/audio/speech`, { method: 'POST', body });
            let json = null;
            try { json = await res.json(); } catch { /* not JSON */ }
            return { status: res.status, json };
        }, config.httpUrl);
        step(
            'cfg_scale=banana returns 400 with a JSON error key',
            bad.status === 400 && typeof bad.json?.error === 'string',
            JSON.stringify(bad),
        );
    } catch (error) {
        step('speech phase aborted', false, error.message);
    }

    return results;
}
