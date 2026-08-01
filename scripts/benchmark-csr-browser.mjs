#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const chromeCandidates = [
    process.env.CHROME_BIN,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
].filter(Boolean);
const chrome = chromeCandidates.find(candidate => fs.existsSync(candidate));
if (!chrome) throw new Error('Chrome/Chromium not found; set CHROME_BIN to run the browser CSR benchmark');

const contentTypes = new Map([
    ['.html', 'text/html; charset=utf-8'],
    ['.mjs', 'text/javascript; charset=utf-8'],
    ['.js', 'text/javascript; charset=utf-8'],
]);
const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const file = path.resolve(root, `.${pathname}`);
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
        response.writeHead(404).end('not found');
        return;
    }
    response.setHeader('content-type', contentTypes.get(path.extname(file)) ?? 'application/octet-stream');
    fs.createReadStream(file).pipe(response);
});

await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
});
const address = server.address();
const url = `http://127.0.0.1:${address.port}/scripts/benchmark-csr-browser.html`;
const args = [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    '--js-flags=--max-old-space-size=256',
    '--virtual-time-budget=30000',
    '--dump-dom',
    url,
];
let stdout = '';
let stderr = '';
const child = spawn(chrome, args, { stdio: ['ignore', 'pipe', 'pipe'] });
child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
const timer = setTimeout(() => child.kill('SIGKILL'), 60_000);
const exitCode = await new Promise(resolve => child.once('exit', resolve));
clearTimeout(timer);
server.close();
const match = stdout.match(/<pre id="result">([\s\S]*?)<\/pre>/);
const result = match?.[1]?.replaceAll('&quot;', '"').replaceAll('&amp;', '&') ?? '';
if (exitCode !== 0 || !result.startsWith('PASS ')) {
    throw new Error(`browser CSR benchmark failed (exit ${exitCode}): ${result || stderr.slice(-2000)}`);
}
console.log(result.slice(5));
