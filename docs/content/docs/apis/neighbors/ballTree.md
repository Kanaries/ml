---
title: BallTree
description: API and practical guide for BallTree in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Neighbors.BallTree

```ts
constructor(
    X: number[][] = [],
    leafSize: number = 40,
    metric: Distance.IDistanceType = 'euclidiean',
    p: number = 2
)
```

### Parameters

- `X` *(number[][])*: training data used to build the tree. Can be provided at
  construction time or via `fit`.
- `leafSize` *(number)*: maximum number of points stored in a leaf. Default is
  `40`.
- `metric` *(Distance.IDistanceType)*: distance function for search. Defaults to
  `'euclidiean'`.
- `p` *(number)*: norm order for Minkowski distance when applicable. Default is
  `2`.

### Algorithm

Ball tree organizes points in hyperspheres. Each node stores a centroid and
radius covering its children. During queries the tree is traversed to prune
branches that are farther than the currently found neighbors, leading to faster
neighbor search than a brute-force approach.

`query(X: number[][], k: number = 1)` returns distances and indices of nearest neighbors.

`queryRadius(X: number[][], r: number, returnDistance = false)` finds neighbors within given radius.

### Example
```ts
const tree = new BallTree(X, 2);
const result = tree.query(X.slice(0, 1), 3);
```

## Practical guide: BallTree in JavaScript and TypeScript

BallTree accelerates nearest-neighbor queries using hierarchical metric-space partitioning.

### When to use BallTree
- You run repeated k-nearest-neighbor lookups on static or slowly changing datasets.
- Brute-force distance scans are too slow at your scale.
- Your metric choice is compatible with tree-based neighbor search.

### Implementation workflow
1. Build the tree from normalized feature vectors.
2. Query nearest points for recommendation, retrieval, or local modeling.
3. Benchmark query latency and rebuild policy as data evolves.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `BallTree JavaScript`
- `BallTree TypeScript`
- `BallTree browser machine learning`
- `@kanaries/ml BallTree`

