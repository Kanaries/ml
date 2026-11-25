// node scripts/e-means/run.mjs DATA_DIRECTORY DATASET [--chrome]
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { benchmark, datasets, decode } from './benchmark.mjs';

const [dir, name, flag] = process.argv.slice(2);
if (!Object.hasOwn(datasets, name)) throw new Error('Choose a dataset from ' + Object.keys(datasets).join(', '));
const [width] = datasets[name];
const train = await readFile(join(dir, name + '-train.f32'));
const test = await readFile(join(dir, name + '-test.f32'));
if (!flag) {
    const rows = buf => decode(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), width);
    const runs = benchmark(name, rows(train), rows(test));
    console.log(JSON.stringify({ name, runtime: process.version, maxRSSKiB: process.resourceUsage().maxRSS, runs }));
} else if (flag === '--chrome') {
    const profile = await mkdtemp(join(tmpdir(), 'emeans-chrome-'));
    const chrome = process.env.CHROME_BIN ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
    const html = `<script type="module">
        import {benchmark, decode} from '/scripts/e-means/benchmark.mjs';
        try {
            const [train,test] = await Promise.all(['/train','/test'].map(async p => decode(await (await fetch(p)).arrayBuffer(), ${width})));
            const runs = benchmark(${JSON.stringify(name)}, train, test);
            await fetch('/result', {method:'POST',body:JSON.stringify({name:${JSON.stringify(name)},runtime:navigator.userAgent,runs})});
        } catch(e) { await fetch('/result',{method:'POST',body:JSON.stringify({error:String(e)})}); }
    </script>`;
    const files = new Map([
        ['/', ['text/html', html]], ['/train', ['application/octet-stream', train]], ['/test', ['application/octet-stream', test]],
        ['/scripts/e-means/benchmark.mjs', ['text/javascript', await readFile(new URL('./benchmark.mjs', import.meta.url))]],
        ['/build/index.mjs', ['text/javascript', await readFile(new URL('../../build/index.mjs', import.meta.url))]],
    ]);
    let finish, fail;
    const result = new Promise((resolve, reject) => { finish = resolve; fail = reject; });
    const server = createServer(async (req, res) => {
        if (req.method === 'POST' && req.url === '/result') {
            const chunks = [];
            for await (const part of req) chunks.push(part);
            res.end(); finish(Buffer.concat(chunks).toString()); return;
        }
        const file = files.get(req.url);
        if (!file) { res.writeHead(404).end(); return; }
        res.setHeader('content-type', file[0]); res.end(file[1]);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const child = spawn(chrome, ['--headless=new', '--disable-gpu', '--no-first-run', '--user-data-dir=' + profile,
        'http://127.0.0.1:' + server.address().port], { stdio: 'ignore' });
    child.once('error', fail);
    const exit = new Promise(resolve => child.once('close', resolve));
    exit.then(code => fail(new Error('Chrome exited: ' + code)));
    const timeout = setTimeout(() => fail(new Error('Browser benchmark timed out')), 600_000);
    try {
        const json = await result;
        if (JSON.parse(json).error) throw new Error(JSON.parse(json).error);
        console.log(json);
    } finally {
        clearTimeout(timeout);
        child.kill(); await exit;
        server.close();
        await rm(profile, { recursive: true, force: true, maxRetries: 3 });
    }
} else throw new Error('Unknown runner option: ' + flag);
