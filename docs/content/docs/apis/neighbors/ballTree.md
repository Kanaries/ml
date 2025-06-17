---
title: BallTree
description: API reference for BallTree
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
