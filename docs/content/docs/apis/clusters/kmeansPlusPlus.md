---
title: kmeansPlusPlus
---

# Clusters.kmeansPlusPlus

```ts
kmeansPlusPlus(
    X: number[][],
    n_clusters: number,
    sampleWeight?: number[],
    randomState: () => number = Math.random
): { centers: number[][]; indices: number[] }
```

This utility initializes cluster centers using the k-means++ strategy.

```ts
const { centers } = kmeansPlusPlus(X, 3);
```
