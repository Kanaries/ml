---
title: MDS
description: API reference for MDS
---

# Manifold.MDS

Multidimensional scaling using classical MDS algorithm.

```ts
interface MDSOptions {
    nComponents?: number;
    dissimilarity?: 'euclidean' | 'precomputed';
}
constructor(options: MDSOptions = {})
```

`fitTransform(data: number[][]): number[][]` computes the embedding and returns it.

`getEmbedding(): number[][]` returns the computed embedding.
