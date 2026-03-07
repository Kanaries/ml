---
title: MDS
description: API and practical guide for MDS in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Manifold.MDS

Multidimensional scaling using classical MDS algorithm.

Classical MDS converts a distance matrix into a centered similarity matrix and
computes its dominant eigenvectors to recover coordinates that preserve the
original pairwise dissimilarities.

```ts
interface MDSOptions {
    nComponents?: number;
    dissimilarity?: 'euclidean' | 'precomputed';
}
constructor(options: MDSOptions = {})
```

### Options
- `nComponents` (number, default `2`): dimension of the embedded space.
- `dissimilarity` (`'euclidean'` | `'precomputed'`, default `'euclidean'`): if
  `'precomputed'`, the input to `fitTransform` should be a distance matrix.

`fitTransform(data: number[][]): number[][]` computes the embedding and returns it.

`getEmbedding(): number[][]` returns the computed embedding.

## Practical guide: MDS in JavaScript and TypeScript

MDS embeds points into lower dimensions while preserving pairwise distances as much as possible.

### When to use MDS
- Distance geometry is more important than original feature axes.
- You need interpretable 2D/3D maps for exploratory analysis.
- You are comparing similarity relationships across entities.

### Implementation workflow
1. Choose or compute a distance matrix aligned with your domain.
2. Fit MDS with target component count for visualization.
3. Validate stress/error and inspect neighborhood preservation quality.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `MDS JavaScript`
- `MDS TypeScript`
- `MDS browser machine learning`
- `@kanaries/ml MDS`

