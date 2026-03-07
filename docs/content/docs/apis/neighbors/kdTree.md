---
title: KDTree
description: API and practical guide for KDTree in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Neighbors.KDTree

```ts
constructor(
    X: number[][] = [],
    leafSize: number = 40,
    metric: Distance.IDistanceType = 'euclidiean',
    p: number = 2
)
```

### Parameters

- `X` *(number[][])*: data used to build the tree. You can also call `fit` later.
- `leafSize` *(number)*: maximum samples per leaf. Default is `40`.
- `metric` *(Distance.IDistanceType)*: distance metric used for queries. Default
  `'euclidiean'`.
- `p` *(number)*: order of the norm when using Minkowski distance. Default `2`.

### Algorithm

KD-tree recursively splits points by dimension. Each internal node stores a
split dimension and value and points to left and right subtrees. During search
the tree is pruned using bounding boxes to efficiently locate nearest
neighbors.

`query(X: number[][], k: number = 1)` returns distances and indices of nearest neighbors.

`queryRadius(X: number[][], r: number, returnDistance = false)` finds neighbors within given radius.

### Example
```ts
const tree = new KDTree(X, 2);
const result = tree.query(X.slice(0, 1), 3);
```

## Practical guide: KDTree in JavaScript and TypeScript

KDTree speeds up nearest-neighbor searches for low-to-medium dimensional numeric feature spaces.

### When to use KDTree
- You need many repeated neighbor lookups with Euclidean-like distances.
- Dataset dimensionality is not too high for kd-tree pruning to remain effective.
- You want faster KNN-style operations in JS services or browser apps.

### Implementation workflow
1. Index feature vectors with KDTree construction.
2. Run neighbor queries and capture distances/indices.
3. Tune leaf/query parameters to balance speed and accuracy.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `KDTree JavaScript`
- `KDTree TypeScript`
- `KDTree browser machine learning`
- `@kanaries/ml KDTree`

