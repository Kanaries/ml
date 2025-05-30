---
title: SpectralEmbedding
---

# Manifold.SpectralEmbedding

Spectral embedding for non-linear dimensionality reduction using the Laplacian Eigenmaps algorithm.

```ts
interface SpectralEmbeddingProps {
    nComponents?: number;
    nNeighbors?: number;
}
constructor(props: SpectralEmbeddingProps = {})
```

### Methods
- `fit(X: number[][]): void`
- `fitTransform(X: number[][]): number[][]`
- `getEmbedding(): number[][]`

### Example
```ts
const se = new SpectralEmbedding({ nComponents: 2, nNeighbors: 5 });
const T = se.fitTransform(X);
```
