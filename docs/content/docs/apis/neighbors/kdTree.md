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

`query(X: number[][], k: number = 1)` returns distances and indices of nearest neighbors.

`queryRadius(X: number[][], r: number, returnDistance = false)` finds neighbors within given radius.

### Example
```ts
const tree = new KDTree(X, 2);
const result = tree.query(X.slice(0, 1), 3);
```
