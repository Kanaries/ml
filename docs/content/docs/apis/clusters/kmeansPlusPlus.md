---
title: kmeansPlusPlus
description: API and practical guide for kmeansPlusPlus in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: kmeansPlusPlus in JavaScript and TypeScript

kmeansPlusPlus provides improved centroid initialization to make KMeans more stable and accurate.

### When to use kmeansPlusPlus
- Random initialization creates unstable cluster assignments.
- You need faster convergence and better centroid quality.
- You run repeated clustering jobs in production pipelines.

### Implementation workflow
1. Initialize centroids with k-means++ strategy before training.
2. Run KMeans fitting with the initialized seeds and compare inertia.
3. Retain the best run by objective score and validation diagnostics.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `kmeansPlusPlus JavaScript`
- `kmeansPlusPlus TypeScript`
- `kmeansPlusPlus browser machine learning`
- `@kanaries/ml kmeansPlusPlus`

