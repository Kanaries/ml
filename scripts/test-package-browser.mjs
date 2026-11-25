import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import path from 'node:path';
import { build } from 'esbuild';
import { chrome } from './chrome.mjs';

const entry = `
import { Clusters, utils } from '@kanaries/ml';

async function check() {
    const model = new Clusters.KMeans({ n_clusters: 2, initCenters: [[0], [8]], max_iter: 2 });
    const labels = model.fitPredict([[0], [1], [7], [8]]);
    if (labels.join(',') !== '0,0,1,1') throw new Error('Unexpected clustering: ' + labels);

    const sum = utils.asyncMode((a, b) => {
        if (!(self instanceof WorkerGlobalScope)) throw new Error('Expected a Web Worker');
        return a + b;
    });
    if (await sum(2, 3) !== 5) throw new Error('Unexpected worker sum');
    const fail = utils.asyncMode(() => Promise.reject(new Error('worker failure')));
    await fail().then(
        () => { throw new Error('Expected a rejected worker promise'); },
        error => { if (error.message !== 'worker failure') throw error; },
    );
}
check().then(() => 'passed', error => String(error)).then(message =>
    fetch('/result?message=' + encodeURIComponent(message), { method: 'POST' }));
`;

export async function checkBrowserConsumer(consumer) {
    const bundle = await build({
        stdin: { contents: entry, resolveDir: consumer },
        platform: 'browser', bundle: true, write: false,
    });

    // Wait for the actual worker reply. A loaded page can still have a worker in flight.
    let report;
    const result = new Promise(resolve => { report = resolve; });
    const server = createServer((req, res) => {
        const url = new URL(req.url, 'http://localhost');
        if (req.method === 'POST' && url.pathname === '/result') {
            res.writeHead(204).end();
            report(url.searchParams.get('message'));
        } else if (url.pathname === '/app.js') {
            res.writeHead(200, { 'Content-Type': 'text/javascript' }).end(bundle.outputFiles[0].contents);
        } else if (url.pathname === '/') {
            res.writeHead(200, { 'Content-Type': 'text/html' }).end('<script src="/app.js"></script>');
        } else {
            res.writeHead(404).end();
        }
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const child = spawn(chrome, [
        '--headless=new', '--disable-gpu', '--no-first-run',
        '--user-data-dir=' + path.join(consumer, 'chrome-profile'),
        'http://127.0.0.1:' + server.address().port,
    ], { stdio: 'ignore', timeout: 30000 });
    const closed = once(child, 'close');
    try {
        const outcome = await Promise.race([result, closed.then(([code, signal]) =>
            `Chrome exited before reporting a result (${signal ?? code})`)]);
        assert.equal(outcome, 'passed');
        console.log('Packed browser consumer predicted and ran a Web Worker in Chrome.');
    } finally {
        child.kill();
        server.close();
        await closed;
    }
}
