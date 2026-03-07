---
title: Manifold
description: Learn how to use Manifold algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [SpectralEmbedding](spectralEmbedding)
- [MDS](MDS)
- [LocallyLinearEmbedding](lle)
- [TSNE](tsne)

## How to use the Manifold module in real projects

The Manifold module helps project high-dimensional data into lower-dimensional embeddings for visualization, exploration, and neighborhood analysis.

### Selection checklist
1. Use t-SNE for local neighborhood visualization and cluster exploration in 2D/3D.
2. Use MDS when preserving pairwise distance structure is the main requirement.
3. Use LocallyLinearEmbedding or SpectralEmbedding when manifold assumptions are meaningful for your data geometry.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `tsne javascript`
- `manifold learning typescript`
- `spectral embedding browser`

### Explore algorithms in this module
- [MDS](MDS)
- [lle](lle)
- [spectralEmbedding](spectralEmbedding)
- [tsne](tsne)

