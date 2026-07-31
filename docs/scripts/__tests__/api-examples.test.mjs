import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import * as ml from '@kanaries/ml';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const API_ROOT = path.resolve(TEST_DIR, '..', '..', 'content', 'docs', 'apis');

async function walk(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const absolute = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(absolute) : [absolute];
  }));
  return files.flat().filter((file) => /\.mdx?$/.test(file) && !/[/\\]index\.(?:md|mdx)$/.test(file));
}

function quickStartCode(source, file) {
  const heading = /^#{2,3} Quick[^\n]*$/m.exec(source);
  assert.ok(heading, `${file} must have a quick-start heading`);
  const fence = /```(?:ts|typescript|js|javascript)\n([\s\S]*?)```/.exec(source.slice(heading.index));
  assert.ok(fence, `${file} must have a JavaScript or TypeScript quick-start block`);
  return fence[1];
}

test('every API page has a copy-paste quick start from package import to visible output', async () => {
  const files = await walk(API_ROOT);
  assert.ok(files.length >= 50);
  for (const file of files) {
    const code = quickStartCode(await fs.readFile(file, 'utf8'), file);
    assert.match(code, /import\s+\{[^}]+\}\s+from\s+'@kanaries\/ml'/, `${file} must import the public package`);
    assert.match(code, /console\.log\s*\(/, `${file} must print a concrete output`);
  }
});

test('every algorithm quick start executes against the published docs dependency', async () => {
  const files = (await walk(API_ROOT)).filter((file) => !file.includes(`${path.sep}utils${path.sep}`));
  for (const file of files) {
    const code = quickStartCode(await fs.readFile(file, 'utf8'), file);
    const compiled = ts.transpileModule(code, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const module = { exports: {} };
    const execute = new Function(
      'require',
      'module',
      'exports',
      'console',
      `return (async () => { ${compiled} })();`,
    );
    await execute(
      (specifier) => {
        assert.equal(specifier, '@kanaries/ml', `${file} imports an unexpected dependency`);
        return ml;
      },
      module,
      module.exports,
      { log() {} },
    );
  }
});
