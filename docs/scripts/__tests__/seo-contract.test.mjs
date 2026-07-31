import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.resolve(TEST_DIR, '..', '..');
const REPO_ROOT = path.resolve(DOCS_ROOT, '..');

function articleWords(source) {
  const article = source.match(/<ToolPageLayout[\s\S]*?>([\s\S]*?)<\/ToolPageLayout>/)?.[1] ?? '';
  return article
    .replace(/<[^>]+>/g, ' ')
    .replace(/\{[^}]+\}/g, ' ')
    .replace(/&\w+;/g, ' ')
    .trim()
    .split(/\s+/).length;
}

const pages = [
  { file: 'app/tools/confusion-matrix-calculator/page.tsx', slug: '/tools/confusion-matrix-calculator', words: 1000 },
  { file: 'app/tools/logistic-regression-calculator/page.tsx', slug: '/tools/logistic-regression-calculator', words: 1000 },
  { file: 'app/playground/pca/page.tsx', slug: '/playground/pca', words: 800 },
  { file: 'app/playground/knn/page.tsx', slug: '/playground/knn', words: 800 },
  { file: 'app/playground/gradient-descent/page.tsx', slug: '/playground/gradient-descent', words: 800 },
  { file: 'app/playground/kmeans/page.tsx', slug: '/playground/kmeans', words: 800 },
  { file: 'app/playground/decision-tree/page.tsx', slug: '/playground/decision-tree', words: 800 },
  { file: 'app/playground/random-forest/page.tsx', slug: '/playground/random-forest', words: 800 },
];

test('all calculator and playground landing pages satisfy the shared SEO template', async () => {
  for (const page of pages) {
    const source = await fs.readFile(path.join(DOCS_ROOT, page.file), 'utf8');
    assert.ok(articleWords(source) >= page.words, `${page.slug} needs at least ${page.words} article words`);
    assert.match(source, /alternates: \{ canonical:/, `${page.slug} needs a canonical URL`);
    assert.match(source, /const faq: ToolFaq\[\]/, `${page.slug} needs visible and structured FAQ content`);
    assert.ok((source.match(/href: '\//g) ?? []).length >= 3, `${page.slug} needs at least three related internal links`);
  }
});

test('P0 pages retain their target titles and differentiating features', async () => {
  const confusion = await fs.readFile(path.join(DOCS_ROOT, pages[0].file), 'utf8');
  const logistic = await fs.readFile(path.join(DOCS_ROOT, pages[1].file), 'utf8');
  assert.match(confusion, /Confusion Matrix Calculator — F1, Precision, Recall \(2-class & Multiclass\)/);
  assert.match(confusion, /F1 score calculator/);
  assert.match(logistic, /Logistic Regression Calculator — Odds Ratios & Decision Boundary, Free/);
  assert.match(logistic, /logistic regression decision boundary/);

  const confusionTool = await fs.readFile(path.join(DOCS_ROOT, 'components/tools/ConfusionMatrixCalculator.tsx'), 'utf8');
  const logisticTool = await fs.readFile(path.join(DOCS_ROOT, 'components/tools/LogisticRegressionCalculator.tsx'), 'utf8');
  for (const tool of [confusionTool, logisticTool]) {
    assert.match(tool, /downloadSvgAsPng/);
    assert.match(tool, /downloadText/);
    assert.match(tool, /<CodeTabs/);
  }
});

test('structured data, discovery, backlinks, and static OG coverage stay wired', async () => {
  const layout = await fs.readFile(path.join(DOCS_ROOT, 'components/tools/ToolPageLayout.tsx'), 'utf8');
  assert.match(layout, /'@type': 'WebApplication'/);
  assert.match(layout, /'@type': 'FAQPage'/);
  assert.match(layout, /Powered by @kanaries\/ml — the scikit-learn-style ML library for JavaScript/);

  const sitemap = await fs.readFile(path.join(DOCS_ROOT, 'app/sitemap.ts'), 'utf8');
  for (const page of pages) assert.ok(sitemap.includes(page.slug), `${page.slug} is missing from sitemap`);

  const readme = await fs.readFile(path.join(REPO_ROOT, 'README.md'), 'utf8');
  const docsHome = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/index.mdx'), 'utf8');
  assert.match(readme, /ml\.kanaries\.net\/tools/);
  assert.match(readme, /ml\.kanaries\.net\/playground/);
  assert.match(docsHome, /\[interactive machine learning tools\]\(\/tools\)/);
  assert.match(docsHome, /\[algorithm playgrounds\]\(\/playground\)/);

  await Promise.all([
    fs.access(path.join(DOCS_ROOT, 'app/playground/opengraph-image.tsx')),
    fs.access(path.join(DOCS_ROOT, 'app/tools/confusion-matrix-calculator/opengraph-image.tsx')),
    fs.access(path.join(DOCS_ROOT, 'app/tools/logistic-regression-calculator/opengraph-image.tsx')),
  ]);
});
