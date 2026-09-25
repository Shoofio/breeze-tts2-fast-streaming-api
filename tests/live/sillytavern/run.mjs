#!/usr/bin/env node
// Composes and runs the SillyTavern live-test phases (tasks.md T010). Exit code 0 only if every
// step across every phase passed.
//
// Usage: node tests/live/sillytavern/run.mjs <health|voices|speech|full> [--skip <phase>] [--record <name>]
//   --skip <phase>    drop one phase from the composed run (T011 uses this to skip `speech` when
//                     testing against the C++ server, which returns 200 for cfg_scale=banana).
//   --record <name>   write specs/003-cpp-compatible-api/research/live-<name>.md instead of
//                     live-<target>.md (T011 uses this to write live-phase0.md).
import { config } from './config.mjs';
import {
    launchBrowser, captureBreezeEvents, writeRecord, openSillyTavern, captureSettings, restoreSettings,
} from './lib.mjs';
import health from './phases/health.mjs';
import voices from './phases/voices.mjs';
import speech from './phases/speech.mjs';
import full from './phases/full.mjs';

const PHASES = { health, voices, speech, full };

// Composition table (tasks.md T010): each CLI target runs these phase modules, in order. Every
// target starts with health, because that phase selects the Breeze provider (run.mjs itself
// navigates to SillyTavern first, below, before any phase runs).
const COMPOSITIONS = {
    health: ['health'],
    voices: ['health', 'voices'],
    speech: ['health', 'speech'],
    full: ['health', 'voices', 'full'],
};

function parseArgs(argv) {
    const [target, ...rest] = argv;
    let skip = null;
    let record = target;
    for (let i = 0; i < rest.length; i += 1) {
        if (rest[i] === '--skip') skip = rest[i += 1];
        else if (rest[i] === '--record') record = rest[i += 1];
    }
    return { target, skip, record };
}

async function main() {
    const { target, skip, record } = parseArgs(process.argv.slice(2));
    const composition = COMPOSITIONS[target];
    if (!composition) {
        console.error(`Usage: node run.mjs <${Object.keys(COMPOSITIONS).join('|')}> [--skip <phase>] [--record <name>]`);
        process.exit(2);
        return;
    }
    const names = skip ? composition.filter((name) => name !== skip) : composition;
    if (skip && names.length === composition.length) {
        console.log(`note: --skip ${skip} had no effect; "${target}" does not include that phase`);
    }

    const { browser, page } = await launchBrowser(config);
    const events = captureBreezeEvents(page);
    // Navigate once here, before any phase, so settings can be snapshotted before the `health`
    // phase's selectBreezeProvider() starts changing them (review pass 1, finding 7). Every phase
    // assumes the page is already loaded.
    await openSillyTavern(page, config);
    const settingsSnapshot = await captureSettings(page);

    const allResults = [];
    try {
        for (const name of names) {
            console.log(`\n=== ${name} ===`);
            const results = await PHASES[name](page, config, events);
            allResults.push(...results.map((r) => ({ ...r, step: `${name}: ${r.step}` })));
        }
    } catch (error) {
        allResults.push({ step: `${target}: run aborted`, ok: false, detail: error.message });
        console.log('run aborted:', error.message);
        console.log('last events:', JSON.stringify(events.slice(-6)));
    } finally {
        // Restore whatever the user had configured, even when a phase failed or threw.
        await restoreSettings(page, settingsSnapshot).catch((error) => {
            console.log('warning: failed to restore settings:', error.message);
        });
        await browser.close();
    }

    const outPath = writeRecord(record, allResults);
    const failed = allResults.filter((r) => !r.ok).length;
    console.log(`\n${allResults.length} checks, ${failed} failed. Record written to ${outPath}`);
    process.exit(failed ? 1 : 0);
}

main();
