---
title: TSNE
description: API reference for TSNE
---

# Manifold.TSNE

T-distributed Stochastic Neighbor Embedding for visualizing high dimensional data.

TSNE models pairwise similarities both in the original space and in the
embedding. It iteratively updates the embedding using gradient descent to
minimize the Kullback–Leibler divergence between these two distributions.

```ts
constructor(options: TSNEOptions = {})
```

### Options
- `nComponents` (number, default `2`): embedding dimensions
- `perplexity` (number, default `30`)
- `learningRate` (number, default `200`)
- `nIter` (number, default `250`)

### Methods
- `fit(X: number[][]): void`
- `fitTransform(X: number[][]): number[][]`
- `getEmbedding(): number[][]`

### Example
```ts
const tsne = new TSNE({ perplexity: 20 });
const Y = tsne.fitTransform(X);
```
