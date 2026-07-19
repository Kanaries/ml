import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { generateAgentDocs } from '../generate-agent-docs.mjs';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.resolve(TEST_DIR, '..', '..');
const CONTENT_ROOT = path.join(DOCS_ROOT, 'content', 'docs');
const SITE_URL = 'https://ml.kanaries.net';

async function withTempDirectory(run) {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'kanaries-agent-docs-'));
  try {
    return await run(tempRoot);
  } finally {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
}

test('generates a Markdown companion and llms.txt entry for every documentation page', async () => {
  await withTempDirectory(async (publicRoot) => {
    const result = await generateAgentDocs({
      contentRoot: CONTENT_ROOT,
      publicRoot,
      siteUrl: SITE_URL,
    });

    assert.ok(result.pageCount > 0);
    assert.equal(result.pages.length, result.pageCount);
    assert.equal(await fs.readFile(path.join(publicRoot, 'llm.txt'), 'utf8'), result.llms);
    assert.equal(await fs.readFile(path.join(publicRoot, 'llms.txt'), 'utf8'), result.llms);
    assert.match(result.llms, /\/api\/agent\/search\?q=\{query\}&limit=\{1-20\}/);

    for (const page of result.pages) {
      const outputPath = path.join(publicRoot, page.outputRelative);
      const markdown = await fs.readFile(outputPath, 'utf8');
      const prose = markdown.replace(/```[\s\S]*?```/g, '');
      assert.match(markdown, /^---\ntitle: /);
      assert.match(markdown, /^# /m);
      assert.ok(result.llms.includes(`${SITE_URL}${page.markdownPath}`), page.markdownPath);
      assert.doesNotMatch(prose, /^import\s/m);
      assert.doesNotMatch(prose, /<[A-Z][A-Za-z0-9]*/);
    }
  });
});

test('preserves code comparisons and provides text fallbacks for interactive components', async () => {
  await withTempDirectory(async (publicRoot) => {
    await generateAgentDocs({ contentRoot: CONTENT_ROOT, publicRoot, siteUrl: SITE_URL });

    const kmeans = await fs.readFile(
      path.join(publicRoot, 'docs', 'apis', 'clusters', 'kmeans.md'),
      'utf8',
    );
    assert.match(kmeans, /Python \(scikit-learn\)/);
    assert.match(kmeans, /from sklearn\.cluster import KMeans/);
    assert.match(kmeans, /import \{ Clusters \} from '@kanaries\/ml'/);

    const clustering = await fs.readFile(
      path.join(publicRoot, 'docs', 'apis', 'clusters', 'index.html.md'),
      'utf8',
    );
    assert.match(clustering, /interactive comparison of clustering algorithms/);
    assert.match(clustering, /\/docs\/apis\/clusters\/kmeans\.md/);
  });
});

test('all generated internal Markdown links resolve to generated files', async () => {
  await withTempDirectory(async (publicRoot) => {
    const result = await generateAgentDocs({ contentRoot: CONTENT_ROOT, publicRoot, siteUrl: SITE_URL });
    const markdownLink = /\]\((\/docs\/[^)#?]+\.md)(?:#[^)]+)?\)/g;

    for (const page of result.pages) {
      const markdown = await fs.readFile(path.join(publicRoot, page.outputRelative), 'utf8');
      for (const match of markdown.matchAll(markdownLink)) {
        await fs.access(path.join(publicRoot, match[1].replace(/^\//, '')));
      }
    }
  });
});

test('fails when an MDX component has no agent renderer', async () => {
  await withTempDirectory(async (tempRoot) => {
    const contentRoot = path.join(tempRoot, 'content');
    const publicRoot = path.join(tempRoot, 'public');
    await fs.mkdir(contentRoot, { recursive: true });
    await fs.writeFile(
      path.join(contentRoot, 'index.mdx'),
      `---\ntitle: Test\ndescription: Test page.\n---\n\n# Test\n\n<UnknownWidget />\n`,
      'utf8',
    );

    await assert.rejects(
      generateAgentDocs({ contentRoot, publicRoot, siteUrl: SITE_URL }),
      /no agent Markdown renderer is registered for <UnknownWidget>/,
    );
  });
});
