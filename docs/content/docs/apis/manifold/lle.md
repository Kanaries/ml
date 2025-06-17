---
title: LocallyLinearEmbedding
description: API reference for LocallyLinearEmbedding
---

# Manifold.LocallyLinearEmbedding

Locally Linear Embedding (LLE) reconstructs each sample from its nearest
neighbors and finds a low‑dimensional representation that preserves these
local relationships.

```ts
constructor(
    nNeighbors: number = 5,
    nComponents: number = 2,
    reg: number = 0.001
)
```

### Parameters
- `nNeighbors` (number, default `5`): how many neighbors to use for the local
  reconstructions.
- `nComponents` (number, default `2`): dimension of the returned embedding.
- `reg` (number, default `0.001`): regularization value added to the covariance
  matrix to ensure numerical stability.

### Methods
- `fit(X: number[][]): void`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`

### Example
```ts
const lle = new LocallyLinearEmbedding(5, 2);
const Y = lle.fitTransform(X);
```
