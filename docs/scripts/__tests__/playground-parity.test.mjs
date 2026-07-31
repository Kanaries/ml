import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { Clusters, Decomposition, Neighbors } from '@kanaries/ml';

function close(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

test('PCA playground engine matches sklearn 1.7.2 on a fixed dataset', () => {
  const X = [
    [2.5, 2.4, 1], [0.5, 0.7, 0.2], [2.2, 2.9, 0.9], [1.9, 2.2, 0.7],
    [3.1, 3, 0.8], [2.3, 2.7, 1.1], [2, 1.6, 0.5], [1, 1.1, 0.3],
  ];
  const model = new Decomposition.PCA(2);
  const projected = model.fitTransform(X);
  const expectedComponents = [
    [0.6723537870829585, 0.6986422746079186, 0.2446208436047036],
    [0.734655876355925, -0.589322389140226, -0.33612477592965895],
  ];
  const expectedVariance = [1.4461971303506673, 0.0639115746766496];
  const expectedProjected = [[0.6817017581082077, 0.11667516150161578], [-2.0463943577749335, -0.08188870892812268]];

  model.getComponents().forEach((row, i) => row.forEach((value, j) => close(value, expectedComponents[i][j])));
  model.getExplainedVariance().forEach((value, i) => close(value, expectedVariance[i]));
  projected.slice(0, 2).forEach((row, i) => row.forEach((value, j) => close(value, expectedProjected[i][j])));
});

test('KNN playground combinations match sklearn 1.7.2 predictions', () => {
  const trainX = [[-2, -1], [-1.5, -0.6], [-1, 0.2], [1, 0.2], [1.5, 0.8], [2, 1.2], [0, 2]];
  const trainY = [0, 0, 0, 1, 1, 1, 1];
  const queries = [[-1.2, -0.2], [1.2, 0.4], [0, 0.8]];
  for (const weight of ['uniform', 'distance']) {
    for (const metric of ['euclidean', 'manhattan']) {
      const model = new Neighbors.KNearestNeighbors(3, weight, metric);
      model.fit(trainX, trainY);
      assert.deepEqual(model.predict(queries), [0, 1, 1]);
    }
  }
});

test('K-Means playground engine matches sklearn 1.7.2 with identical centers', () => {
  const X = [[-2, -1.8], [-1.8, -2.2], [-2.2, -2], [1.8, 2], [2.2, 1.9], [2, 2.3], [0, 3], [0.2, 2.8]];
  const initial = [[-2, -1.8], [1.8, 2], [0, 3]];
  const model = new Clusters.KMeans(3, 1e-4, initial, 30);
  assert.deepEqual(model.fitPredict(X), [0, 0, 0, 1, 1, 1, 2, 2]);
  const expectedCenters = [[-2, -2], [2, 2.0666666666666664], [0.1, 2.9]];
  model.getCentroids().forEach((row, i) => row.forEach((value, j) => close(value, expectedCenters[i][j])));
  close(model.getInertia(), 0.3666666666666669);
});

test('all P1 and P2 playground pages keep the shared SEO contract', async () => {
  for (const slug of ['pca', 'knn', 'gradient-descent', 'kmeans', 'decision-tree', 'random-forest']) {
    const source = await readFile(new URL(`../../app/playground/${slug}/page.tsx`, import.meta.url), 'utf8');
    const article = source.match(/<ToolPageLayout[\s\S]*?>([\s\S]*?)<\/ToolPageLayout>/)?.[1] ?? '';
    const words = article.replace(/<[^>]+>/g, ' ').replace(/\{[^}]+\}/g, ' ').replace(/&\w+;/g, ' ').trim().split(/\s+/).length;
    assert.ok(words >= 800, `${slug} content should contain at least 800 words, found ${words}`);
    assert.match(source, /alternates: \{ canonical:/);
    assert.match(source, /const faq: ToolFaq\[\]/);
    assert.match(source, /sectionName="Playground"/);
    assert.ok((source.match(/href: '\//g) ?? []).length >= 3, `${slug} should have at least three related internal links`);
  }
});
