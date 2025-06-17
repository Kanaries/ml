---
title: KDTree
description: API reference for KDTree
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
