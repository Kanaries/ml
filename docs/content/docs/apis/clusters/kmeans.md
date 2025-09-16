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