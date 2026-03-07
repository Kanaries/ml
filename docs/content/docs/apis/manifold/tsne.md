---
title: TSNE
description: API and practical guide for TSNE in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: TSNE in JavaScript and TypeScript

TSNE creates low-dimensional visualizations that preserve local neighborhoods in high-dimensional data.

### When to use TSNE
- You need exploratory plots to inspect cluster tendencies or anomalies.
- Local similarity is more important than global distance preservation.
- You can afford iterative optimization for offline analysis workflows.

### Implementation workflow
1. Normalize inputs and optionally reduce dimensions with PCA first.
2. Fit TSNE with tuned perplexity and iteration settings.
3. Interpret clusters carefully and validate findings with quantitative checks.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `TSNE JavaScript`
- `TSNE TypeScript`
- `TSNE browser machine learning`
- `@kanaries/ml TSNE`

