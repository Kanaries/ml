---
title: SpectralEmbedding
description: API reference for SpectralEmbedding
---

# Manifold.SpectralEmbedding

Spectral embedding for non-linear dimensionality reduction using the Laplacian Eigenmaps algorithm.

The algorithm builds a nearest‑neighbor graph from the data, computes the
normalized graph Laplacian and uses its leading eigenvectors (except for the
trivial one) as the embedding coordinates.

```ts
interface SpectralEmbeddingProps {
    nComponents?: number;
    nNeighbors?: number;
}
constructor(props: SpectralEmbeddingProps = {})
```

### Parameters
- `nComponents` (number, default `2`): number of embedding dimensions.
- `nNeighbors` (number, default `10`): how many neighbors are connected in the
  affinity graph.

### Methods
- `fit(X: number[][]): void`
- `fitTransform(X: number[][]): number[][]`
- `getEmbedding(): number[][]`

### Example
```ts
const se = new SpectralEmbedding({ nComponents: 2, nNeighbors: 5 });
const T = se.fitTransform(X);
```
