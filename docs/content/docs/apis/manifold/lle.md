---
title: LocallyLinearEmbedding
description: API reference for LocallyLinearEmbedding
---

# Manifold.LocallyLinearEmbedding

```ts
constructor(
    nNeighbors: number = 5,
    nComponents: number = 2,
    reg: number = 0.001
)
```

### Methods
- `fit(X: number[][]): void`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`

### Example
```ts
const lle = new LocallyLinearEmbedding(5, 2);
const Y = lle.fitTransform(X);
```
