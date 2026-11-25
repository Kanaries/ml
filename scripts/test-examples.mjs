import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { preview } from 'vite';
import { chrome } from './chrome.mjs';

const server = await preview({
    configFile: 'examples/vite.config.ts',
    preview: { host: '127.0.0.1', port: 0, open: false },
});
const origin = server.resolvedUrls.local[0];
const profile = await mkdtemp(path.join(tmpdir(), 'ml-examples-chrome-'));

async function render(page) {
    const { stdout } = await promisify(execFile)(chrome, [
        '--headless=new', '--disable-gpu', '--no-first-run', '--user-data-dir=' + profile,
        '--dump-dom', '--virtual-time-budget=10000', new URL(page, origin).href,
    ], { timeout: 30000, maxBuffer: 4 * 1024 * 1024 });
    return stdout;
}

try {
    const gallery = await render('index.html');
    assert.match(gallery, /<canvas\b/, 'Expected the K-Means plot to render');
    const comparison = await render('e-means.html');
    for (const label of ['E-Means', 'Lloyd', 'MiniBatchKMeans']) {
        assert.ok(comparison.match(new RegExp('<td>' + label + '</td><td>\\d+\\.\\d{3}</td>')),
            'Expected a finite held-out MSE for ' + label);
    }
    console.log('Both built examples rendered in Chrome.');
} finally {
    await server.close();
    await rm(profile, { recursive: true, force: true });
}
