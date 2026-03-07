---
title: SpectralEmbedding
description: API and practical guide for SpectralEmbedding in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: SpectralEmbedding in JavaScript and TypeScript

SpectralEmbedding uses graph Laplacian eigenvectors to reveal manifold and community structure in data.

### When to use SpectralEmbedding
- Graph or affinity relationships are central to your dataset.
- You need embeddings suitable for downstream clustering.
- Non-linear structure is not captured by purely linear projections.

### Implementation workflow
1. Construct an affinity graph with an appropriate similarity metric.
2. Fit SpectralEmbedding and evaluate resulting cluster separability.
3. Tune neighborhood/affinity parameters for stable embeddings.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `SpectralEmbedding JavaScript`
- `SpectralEmbedding TypeScript`
- `SpectralEmbedding browser machine learning`
- `@kanaries/ml SpectralEmbedding`

