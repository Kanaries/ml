---
title: LocallyLinearEmbedding
description: API and practical guide for LocallyLinearEmbedding in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: LocallyLinearEmbedding in JavaScript and TypeScript

LocallyLinearEmbedding captures manifold structure by preserving local linear neighborhoods in a low-dimensional embedding.

### When to use LocallyLinearEmbedding
- Data lies on a non-linear manifold with meaningful local geometry.
- You want neighborhood-preserving visualization or preprocessing.
- Linear projection methods (like PCA) lose important local structure.

### Implementation workflow
1. Select neighborhood size based on expected local manifold smoothness.
2. Fit and inspect embedding quality visually and with neighborhood metrics.
3. Tune neighbors/components to balance stability and detail.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LocallyLinearEmbedding JavaScript`
- `LocallyLinearEmbedding TypeScript`
- `LocallyLinearEmbedding browser machine learning`
- `@kanaries/ml LocallyLinearEmbedding`

