// Build a self-contained demo for the separately deployed documentation site.
import { build } from 'vite';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../../', import.meta.url);
const template = await readFile(new URL('examples/e-means.html', root), 'utf8');
const [{ output }] = await build({
    configFile: false,
    build: {
        write: false,
        lib: { entry: fileURLToPath(new URL('examples/e-means.ts', root)), name: 'EMeansExample', formats: ['iife'] },
    },
});
const license = await readFile(new URL('LICENSE', root), 'utf8');
const html = template.replace('<script type="module" src="./e-means.ts"></script>',
    () => '<!--\n' + license + '\n-->\n<script>' + output[0].code + '</script>');
await writeFile(new URL('docs/public/e-means-example.html', root), html);
