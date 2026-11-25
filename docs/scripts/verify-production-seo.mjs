#!/usr/bin/env node

import assert from 'node:assert/strict';

const baseUrl = (process.env.SEO_BASE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const canonicalBaseUrl = (process.env.SEO_CANONICAL_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

const hubPages = ['/tools', '/playground'];

const landingPages = [
  { path: '/tools/outlier-calculator', title: 'Outlier Calculator — Free CSV Tool' },
  { path: '/tools/polynomial-regression-calculator', title: 'Polynomial Regression Calculator — Free CSV Tool' },
  { path: '/tools/customer-segmentation', title: 'Customer Segmentation Tool — Free CSV Tool' },

  {
    path: '/tools/confusion-matrix-calculator',
    title: 'Confusion Matrix Calculator — F1, Precision, Recall (2-class & Multiclass)',
  },
  {
    path: '/tools/logistic-regression-calculator',
    title: 'Logistic Regression Calculator — Odds Ratios & Decision Boundary, Free',
  },
  { path: '/playground/pca' },
  { path: '/playground/knn' },
  { path: '/playground/gradient-descent' },
  { path: '/playground/kmeans' },
  { path: '/playground/decision-tree' },
  { path: '/playground/random-forest' },
];

const contentPages = [
  '/docs',
  '/docs/apis',
  '/docs/apis/pipeline',
  '/docs/apis/serialization',
  '/docs/apis/mixture/gaussian-mixture',
  '/docs/apis/discriminant-analysis/lda',
  '/docs/apis/calibration',
  '/docs/apis/datasets',
  '/docs/apis/data',
  '/docs/apis/clusters',
  '/docs/apis/clusters/kmeans',
  '/docs/apis/ensemble/iforest',
  '/docs/apis/decomposition/nmf',
  '/docs/apis/gaussian_process/classifier',
  '/docs/guides/javascript-ml-libraries-comparison',
  '/docs/guides/edge-and-web-workers-ml',
  '/docs/guides/migrating-sklearn-pipeline-to-typescript',
  '/docs/guides/find-outliers-csv',
  '/docs/guides/outlier-methods',
  '/docs/guides/choose-polynomial-degree',
  '/docs/guides/outliers-regression',
  '/docs/guides/customer-segmentation-csv',
  '/docs/guides/rfm-vs-kmeans',
  '/docs/guides/isolation-forest',
  '/docs/sklearn-equivalents',
];
const discoveryFiles = ['/robots.txt', '/sitemap.xml', '/llms.txt', '/llms-full.txt'];
const expectedRoutes = [...hubPages, ...landingPages.map((page) => page.path), ...contentPages];

async function requestWithMeta(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: { 'user-agent': '@kanaries/ml production SEO verifier' },
    redirect: options.redirect ?? 'follow',
  });
  const body = await response.text();
  return { status: response.status, headers: response.headers, body };
}

async function request(path) {
  const result = await requestWithMeta(path);
  assert.equal(result.status, 200, `${path} returned HTTP ${result.status}`);
  return result.body;
}

function canonicalFrom(html) {
  return html.match(/<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)/i)?.[1]
    ?? html.match(/<link[^>]+href=["']([^"']+)["'][^>]+rel=["']canonical["']/i)?.[1];
}

function decodeHtml(value) {
  return value
    .replaceAll('&amp;', '&')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>');
}

function titleFrom(html) {
  const title = html.match(/<title>([\s\S]*?)<\/title>/i)?.[1];
  return title ? decodeHtml(title) : undefined;
}

function h1CountFrom(html) {
  return (html.match(/<h1\b/gi) ?? []).length;
}

function hasMeta(html, attrName, attrValue) {
  const pattern = new RegExp(
    `<meta[^>]+${attrName}=["']${attrValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}["'][^>]+content=["'][^"']+["']`,
    'i',
  );
  const reverse = new RegExp(
    `<meta[^>]+content=["'][^"']+["'][^>]+${attrName}=["']${attrValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}["']`,
    'i',
  );
  return pattern.test(html) || reverse.test(html);
}

function assertBrandHygiene(path, title, maxLen) {
  assert.ok(title, `${path} is missing a <title>`);
  const brandCount = (title.match(/@kanaries\/ml/gi) ?? []).length;
  assert.ok(brandCount <= 1, `${path} duplicates @kanaries/ml in title: "${title}"`);
  if (maxLen) {
    assert.ok(title.length <= maxLen, `${path} title exceeds ${maxLen} chars (${title.length}): "${title}"`);
  }
}

function assertSocialAndHeadingHygiene(path, html) {
  assert.equal(h1CountFrom(html), 1, `${path} should render exactly one <h1>, got ${h1CountFrom(html)}`);
  assert.ok(hasMeta(html, 'property', 'og:image'), `${path} is missing og:image`);
  assert.ok(hasMeta(html, 'name', 'twitter:image'), `${path} is missing twitter:image`);
}

function jsonLdTypes(html) {
  const types = new Set();
  for (const match of html.matchAll(/<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    const value = JSON.parse(match[1].replaceAll('&quot;', '"'));
    const visit = (node) => {
      if (Array.isArray(node)) return node.forEach(visit);
      if (!node || typeof node !== 'object') return;
      if (typeof node['@type'] === 'string') types.add(node['@type']);
      else if (Array.isArray(node['@type'])) node['@type'].forEach((t) => types.add(t));
      Object.values(node).forEach(visit);
    };
    visit(value);
  }
  return types;
}

async function verifyHomePage() {
  const html = await request('/');
  assert.equal(canonicalFrom(html), canonicalBaseUrl, '/ canonical is incorrect');
  assertBrandHygiene('/', titleFrom(html), 65);
  assertSocialAndHeadingHygiene('/', html);

  const schemaTypes = jsonLdTypes(html);
  assert.ok(schemaTypes.has('WebSite'), '/ is missing SSR WebSite JSON-LD');
  assert.ok(schemaTypes.has('SoftwareSourceCode'), '/ is missing SSR SoftwareSourceCode JSON-LD');
  assert.ok(schemaTypes.has('FAQPage'), '/ is missing SSR FAQPage JSON-LD');

  assert.match(html, /href=["']\/tools\/[^"']+/, '/ should link directly to interactive tools');
  assert.match(html, /href=["']\/playground\/[^"']+/, '/ should link directly to interactive playgrounds');
  assert.match(html, /href=["']\/docs\/guides\/[^"']+/, '/ should link directly to practical guides');
}

async function verifyHubPage(path) {
  const html = await request(path);
  assert.equal(canonicalFrom(html), `${canonicalBaseUrl}${path}`, `${path} canonical is incorrect`);
  assertBrandHygiene(path, titleFrom(html), 65);
  assertSocialAndHeadingHygiene(path, html);

  const schemaTypes = jsonLdTypes(html);
  assert.ok(schemaTypes.has('CollectionPage'), `${path} is missing CollectionPage JSON-LD`);
  assert.ok(schemaTypes.has('ItemList'), `${path} is missing ItemList JSON-LD`);
  assert.ok(schemaTypes.has('BreadcrumbList'), `${path} is missing BreadcrumbList JSON-LD`);
}

async function verifyLandingPage(page) {
  const html = await request(page.path);
  assert.equal(canonicalFrom(html), `${canonicalBaseUrl}${page.path}`, `${page.path} canonical is incorrect`);
  const title = titleFrom(html);
  if (page.title) assert.equal(title, page.title, `${page.path} title is incorrect`);
  assertBrandHygiene(page.path, title);
  assertSocialAndHeadingHygiene(page.path, html);

  const schemaTypes = jsonLdTypes(html);
  assert.ok(schemaTypes.has('WebApplication'), `${page.path} is missing WebApplication JSON-LD`);
  assert.ok(schemaTypes.has('FAQPage'), `${page.path} is missing FAQPage JSON-LD`);
  assert.ok(schemaTypes.has('BreadcrumbList'), `${page.path} is missing BreadcrumbList JSON-LD`);
  assert.match(html, /Powered by @kanaries\/ml/, `${page.path} is missing the library CTA`);

  const internalLinks = new Set(
    [...html.matchAll(/href=["'](\/(?:tools|playground|docs)\/[^"'#?]+)/g)].map((match) => match[1]),
  );
  assert.ok(internalLinks.size >= 3, `${page.path} exposes fewer than three internal links`);
}

async function verifyContentPage(path) {
  const html = await request(path);
  assert.equal(canonicalFrom(html), `${canonicalBaseUrl}${path}`, `${path} canonical is incorrect`);
  assertBrandHygiene(path, titleFrom(html), 65);
  assertSocialAndHeadingHygiene(path, html);

  const schemaTypes = jsonLdTypes(html);
  assert.ok(schemaTypes.has('TechArticle'), `${path} is missing TechArticle JSON-LD`);
  assert.ok(schemaTypes.has('BreadcrumbList'), `${path} is missing BreadcrumbList JSON-LD`);
}

async function main() {
  await verifyHomePage();
  await Promise.all(hubPages.map(verifyHubPage));
  await Promise.all(landingPages.map(verifyLandingPage));
  await Promise.all(contentPages.map(verifyContentPage));

  const [robotsRes, sitemapRes, llmsRes, llmsFullRes] = await Promise.all(
    discoveryFiles.map((path) => requestWithMeta(path)),
  );
  for (const [idx, res] of [robotsRes, sitemapRes, llmsRes, llmsFullRes].entries()) {
    assert.equal(res.status, 200, `${discoveryFiles[idx]} returned HTTP ${res.status}`);
  }
  const robots = robotsRes.body;
  const sitemap = sitemapRes.body;
  const llms = llmsRes.body;
  const llmsFull = llmsFullRes.body;

  assert.match(robots, new RegExp(`Sitemap: ${canonicalBaseUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/sitemap\\.xml`));
  for (const path of expectedRoutes) {
    assert.ok(sitemap.includes(`${canonicalBaseUrl}${path}`), `${path} is missing from sitemap.xml`);
  }
  const locCount = (sitemap.match(/<loc>/g) ?? []).length;
  const lastmodCount = (sitemap.match(/<lastmod>/g) ?? []).length;
  assert.ok(locCount > 100, `sitemap.xml has unexpectedly few URLs (${locCount})`);
  assert.equal(lastmodCount, locCount, `sitemap.xml lastmod count (${lastmodCount}) must match loc count (${locCount})`);

  assert.match(llmsRes.headers.get('x-robots-tag') ?? '', /noindex/i, '/llms.txt must send X-Robots-Tag: noindex');
  assert.match(llmsFullRes.headers.get('x-robots-tag') ?? '', /noindex/i, '/llms-full.txt must send X-Robots-Tag: noindex');
  assert.ok(llms.includes(`${canonicalBaseUrl}/docs/sklearn-equivalents.md`), 'llms.txt is missing the sklearn mapping');
  assert.ok(llms.includes(`${canonicalBaseUrl}/docs/guides/isolation-forest.md`), 'llms.txt is missing the Isolation Forest guide');
  assert.ok(llmsFull.length > 200_000, `llms-full.txt is unexpectedly small (${llmsFull.length} bytes)`);

  const llmAlias = await requestWithMeta('/llm.txt', { redirect: 'manual' });
  assert.ok(
    [301, 308].includes(llmAlias.status) && (llmAlias.headers.get('location') ?? '').endsWith('/llms.txt'),
    `/llm.txt should permanently redirect to /llms.txt (got ${llmAlias.status} -> ${llmAlias.headers.get('location')})`,
  );

  console.log(`Production SEO verification passed for ${baseUrl}`);
  console.log(
    `1 homepage, ${hubPages.length} hub pages, ${landingPages.length} landing pages, ${contentPages.length} content pages, and ${discoveryFiles.length} discovery files checked.`,
  );
}

main().catch((error) => {
  console.error(`Production SEO verification failed for ${baseUrl}`);
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
