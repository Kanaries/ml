#!/usr/bin/env node

import assert from 'node:assert/strict';

const baseUrl = (process.env.SEO_BASE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const canonicalBaseUrl = (process.env.SEO_CANONICAL_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

const landingPages = [
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

const contentPages = ['/docs/guides/isolation-forest', '/docs/sklearn-equivalents'];
const discoveryFiles = ['/robots.txt', '/sitemap.xml', '/llms.txt', '/llms-full.txt'];
const expectedRoutes = [...landingPages.map((page) => page.path), ...contentPages];

async function request(path) {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: { 'user-agent': '@kanaries/ml production SEO verifier' },
    redirect: 'follow',
  });
  const body = await response.text();
  assert.equal(response.status, 200, `${path} returned HTTP ${response.status}`);
  return body;
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

function jsonLdTypes(html) {
  const types = new Set();
  for (const match of html.matchAll(/<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    const value = JSON.parse(match[1].replaceAll('&quot;', '"'));
    const visit = (node) => {
      if (Array.isArray(node)) return node.forEach(visit);
      if (!node || typeof node !== 'object') return;
      if (typeof node['@type'] === 'string') types.add(node['@type']);
      Object.values(node).forEach(visit);
    };
    visit(value);
  }
  return types;
}

async function verifyLandingPage(page) {
  const html = await request(page.path);
  assert.equal(canonicalFrom(html), `${canonicalBaseUrl}${page.path}`, `${page.path} canonical is incorrect`);
  if (page.title) assert.equal(titleFrom(html), page.title, `${page.path} title is incorrect`);

  const schemaTypes = jsonLdTypes(html);
  assert.ok(schemaTypes.has('WebApplication'), `${page.path} is missing WebApplication JSON-LD`);
  assert.ok(schemaTypes.has('FAQPage'), `${page.path} is missing FAQPage JSON-LD`);
  assert.match(html, /Powered by @kanaries\/ml/, `${page.path} is missing the library CTA`);

  const internalLinks = new Set(
    [...html.matchAll(/href=["'](\/(?:tools|playground|docs)\/[^"'#?]+)/g)].map((match) => match[1]),
  );
  assert.ok(internalLinks.size >= 3, `${page.path} exposes fewer than three internal links`);
}

async function main() {
  await Promise.all(landingPages.map(verifyLandingPage));

  for (const path of contentPages) {
    const html = await request(path);
    assert.equal(canonicalFrom(html), `${canonicalBaseUrl}${path}`, `${path} canonical is incorrect`);
  }

  const [robots, sitemap, llms, llmsFull] = await Promise.all(discoveryFiles.map(request));
  assert.match(robots, new RegExp(`Sitemap: ${canonicalBaseUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/sitemap\\.xml`));
  for (const path of expectedRoutes) {
    assert.ok(sitemap.includes(`${canonicalBaseUrl}${path}`), `${path} is missing from sitemap.xml`);
  }
  assert.ok(llms.includes(`${canonicalBaseUrl}/docs/sklearn-equivalents.md`), 'llms.txt is missing the sklearn mapping');
  assert.ok(llms.includes(`${canonicalBaseUrl}/docs/guides/isolation-forest.md`), 'llms.txt is missing the Isolation Forest guide');
  assert.ok(llmsFull.length > 200_000, `llms-full.txt is unexpectedly small (${llmsFull.length} bytes)`);

  console.log(`Production SEO verification passed for ${baseUrl}`);
  console.log(`${landingPages.length} landing pages, ${contentPages.length} content pages, and ${discoveryFiles.length} discovery files checked.`);
}

main().catch((error) => {
  console.error(`Production SEO verification failed for ${baseUrl}`);
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
