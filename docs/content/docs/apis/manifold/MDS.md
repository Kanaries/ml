---
title: MDS
description: API reference for MDS
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
