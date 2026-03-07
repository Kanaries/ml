---
title: K-Means Clustering (KMeans) – @kanaries/ml Algorithm Guide
description: Learn how to apply the K-Means clustering algorithm with @kanaries/ml, including parameter definitions, TypeScript API usage, and examples for segmenting datasets in modern web apps.
---

# K-Means Clustering (Clusters.KMeans)

K-Means clustering partitions datasets into a chosen number of groups by minimizing within-cluster variance. Use the `Clusters.KMeans` implementation in @kanaries/ml to build fast, client-side segmentation pipelines for browser or Node.js applications.

```ts
constructor (n_clusters: number = 2, opt_ratio: number = 0.05, initCenters?: number[][], max_iter: number = 30)
```

| props name | type | default value |
|-|-|-|
| n_clusters | number | 2 |
| opt_ratio | number | 0.05 |
| initCenters | number[][] | undefined |
| max_iter | number | 30 |


```js
const X = [
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [1, 1],
];
const sampleWeights = [3, 1, 1, 3];
const initCenters = [[0, 0], [1, 1]];

const kmeans = new KMeans(2, 0.05, initCenters);

const result = kmeans.fitPredict(X, sampleWeights);

```

## Practical guide: K-Means Clustering (KMeans) in JavaScript and TypeScript

KMeans partitions data into k centroid-based groups for segmentation and coarse structure discovery.

### When to use K-Means Clustering (KMeans)
- You can estimate a reasonable cluster count from domain knowledge.
- Clusters are roughly spherical in normalized feature space.
- You need fast, repeatable clustering for product analytics workflows.

### Implementation workflow
1. Normalize numeric features and choose an initial value of `k`.
2. Train with multiple seeds when possible to reduce local-minimum sensitivity.
3. Validate cluster usefulness with silhouette score and business outcomes.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `K-Means Clustering (KMeans) JavaScript`
- `K-Means Clustering (KMeans) TypeScript`
- `K-Means Clustering (KMeans) browser machine learning`
- `@kanaries/ml K-Means Clustering (KMeans)`

