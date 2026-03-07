---
title: Decomposition
description: Learn how to use Decomposition algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [PCA](pca)
- [TruncatedSVD](truncatedSVD)
- [SparsePCA](sparsePCA)

## How to use the Decomposition module in real projects

The Decomposition module reduces feature dimensions while preserving useful signal for visualization, denoising, and faster downstream models.

### Selection checklist
1. Use PCA for dense continuous tabular data when variance preservation is the goal.
2. Use TruncatedSVD for sparse matrices such as text vectors or recommendation matrices.
3. Use SparsePCA when you need components with feature-level sparsity for interpretability.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `pca javascript`
- `truncated svd typescript`
- `dimensionality reduction browser ml`

### Explore algorithms in this module
- [pca](pca)
- [sparsePCA](sparsePCA)
- [truncatedSVD](truncatedSVD)

