import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.resolve(TEST_DIR, '..', '..');

const clusterPages = [
  { file: 'index.mdx', pattern: /<ClusteringComparison \/>/ },
  { file: 'kmeans.mdx', pattern: /<KMeansPlayground \/>/ },
  { file: 'dbscan.mdx', pattern: /<ClusteringPlayground algorithm="dbscan" \/>/ },
  { file: 'hdbscan.mdx', pattern: /<ClusteringPlayground algorithm="hdbscan" \/>/ },
  { file: 'meanShift.mdx', pattern: /<ClusteringPlayground algorithm="meanShift" \/>/ },
  { file: 'optics.mdx', pattern: /<ClusteringPlayground algorithm="optics" \/>/ },
  { file: 'kmeansPlusPlus.mdx', pattern: /<ClusteringPlayground algorithm="kmeansPlusPlus" \/>/ },
  { file: 'advancedClustering.mdx', pattern: /<ClusteringPlayground algorithm="advanced" \/>/ },
];

test('every clustering documentation page embeds an online interactive module', async () => {
  for (const page of clusterPages) {
    const source = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/apis/clusters', page.file), 'utf8');
    assert.match(source, page.pattern, `${page.file} is missing its clustering interaction`);
  }
});

test('every tree documentation page embeds the shared live tree module', async () => {
  for (const file of [
    'index.mdx',
    'decisionTreeClassifier.mdx',
    'decisionTreeRegressor.mdx',
    'extraTreeClassifier.mdx',
    'extraTreeRegressor.mdx',
  ]) {
    const source = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/apis/tree', file), 'utf8');
    assert.match(source, /<DecisionTreePlayground task=/, `${file} is missing its tree interaction`);
  }
});

test('the shared clustering playground fits every documented algorithm with @kanaries/ml', async () => {
  const source = await fs.readFile(path.join(DOCS_ROOT, 'components/clusteringPlayground.tsx'), 'utf8');
  for (const api of [
    'Clusters.DBScan',
    'Clusters.HDBScan',
    'Clusters.MeanShift',
    'Clusters.OPTICS',
    'Clusters.kmeansPlusPlus',
    'Clusters.Birch',
    'Clusters.AffinityPropagation',
    'Clusters.BisectingKMeans',
  ]) {
    assert.ok(source.includes(api), `${api} is not wired into the interactive component`);
  }
  assert.match(source, /onClick={handlePlotClick}/, 'cluster plot should accept user-added observations');
  assert.match(source, /aria-label="Add an observation by coordinates"/, 'cluster plot should provide a keyboard-accessible coordinate input');
  assert.match(source, /aria-live="polite"/, 'cluster metrics should announce live model changes');
  assert.match(source, /function clusterColor\(label/, 'cluster colors should remain distinct beyond the fixed palette');
  assert.match(source, /'Seed regions'/, 'k-means++ output should distinguish seed regions from fitted clusters');
  assert.match(source, /createRandom\(parameters\.modelSeed\)/, 'randomized seeding should remain reproducible');
});
