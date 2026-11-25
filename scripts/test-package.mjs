import { execFileSync } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const consumer = await mkdtemp(path.join(tmpdir(), 'kanaries-ml-consumer-'));
const require = createRequire(import.meta.url);

try {
    const tarball = execFileSync('npm', ['pack', '--silent', '--ignore-scripts', '--pack-destination', consumer], {
        cwd: root, encoding: 'utf8',
    }).trim();
    await writeFile(path.join(consumer, 'package.json'), JSON.stringify({ private: true }));
    execFileSync('npm', ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', './' + tarball], {
        cwd: consumer, stdio: 'inherit',
    });

    const checks = `
async function checkWorker() {
    const sum = utils.asyncMode((a, b) => a + b);
    assert.equal(await sum(2, 3), 5);
    const fail = utils.asyncMode(() => { throw new Error('worker failure'); });
    await assert.rejects(fail(), /worker failure/);
}
checkWorker().catch(error => { console.error(error); process.exitCode = 1; });
`;
    for (const [file, imports] of [
        ['consumer.mjs', "import assert from 'node:assert/strict'; import { utils } from '@kanaries/ml';"],
        ['consumer.cjs', "const assert = require('node:assert/strict'); const { utils } = require('@kanaries/ml');"],
    ]) {
        await writeFile(path.join(consumer, file), imports + checks);
        execFileSync(process.execPath, [file], { cwd: consumer, stdio: 'inherit', timeout: 15000 });
    }

    await writeFile(path.join(consumer, 'consumer.mts'), `
import { utils } from '@kanaries/ml';
const add = utils.asyncMode((a: number, b: number) => a + b);
const result: Promise<number> = add(2, 3);
// @ts-expect-error The worker preserves the argument types.
add('2', 3);
`);
    execFileSync(process.execPath, [require.resolve('typescript/bin/tsc'), '--noEmit', '--strict',
        '--skipLibCheck', '--module', 'NodeNext', 'consumer.mts'], { cwd: consumer, stdio: 'inherit' });
    console.log('Packed ESM, CommonJS, and TypeScript consumers passed.');
    if (process.argv.includes('--browser')) {
        const { checkBrowserConsumer } = await import('./test-package-browser.mjs');
        await checkBrowserConsumer(consumer);
    }
} finally {
    await rm(consumer, { recursive: true, force: true, maxRetries: 4 });
}
