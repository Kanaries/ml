---
title: Dimensionality Reduction in JavaScript with @kanaries/ml
description: Explore PCA, Kernel PCA, FastICA, NMF, Incremental PCA, Sparse PCA, and Truncated SVD in JavaScript and TypeScript with @kanaries/ml.
---

# Dimensionality Reduction in JavaScript

## Module overview

The Decomposition module reduces feature dimensions while preserving useful signal for visualization, denoising, compression, and downstream modeling. These algorithms are often used before clustering, classification, or charting when raw feature spaces are too large or too noisy.

This module is a strong fit when:

- you need lower-dimensional inputs for charts or interactive analysis
- you want to compress features before training another model
- you need different strategies for dense versus sparse matrices

## JavaScript implementation

`@kanaries/ml` provides decomposition algorithms directly in JavaScript and TypeScript so frontend and full-stack teams can run feature compression in the same runtime that already owns embeddings, visualizations, or data pipelines. This is especially useful when dimensionality reduction is part of a browser-based analytics experience or a Node.js preprocessing step.

If someone searches for "PCA in JavaScript" or "Truncated SVD in TypeScript", this module is the right entry point.

## Quick navigation

- [PCA](pca): dense feature compression and variance-preserving projections
- [Sparse PCA](sparsePCA): more interpretable sparse components
- [Truncated SVD](truncatedSVD): latent factors for sparse or vectorized inputs
- [Kernel PCA](kernelPCA): nonlinear kernel embeddings
- [FastICA](fastICA): independent signal separation
- [NMF](nmf): non-negative, parts-based latent factors
- [Incremental PCA](incrementalPCA): batch-wise, memory-aware PCA

## Detailed module guide

### How to choose an algorithm

1. Use [PCA](pca) for dense numeric features when variance preservation matters most.
2. Use [Truncated SVD](truncatedSVD) for sparse matrices such as TF-IDF or one-hot vectors.
3. Use [Sparse PCA](sparsePCA) when component interpretability matters alongside dimensionality reduction.
4. Use [Kernel PCA](kernelPCA) for nonlinear structure, [FastICA](fastICA) for independent sources, [NMF](nmf) for non-negative additive factors, or [Incremental PCA](incrementalPCA) for streamed batches.

### JavaScript deployment notes

- Standardize dense continuous features before PCA-style methods when appropriate.
- Use decomposition to simplify data before charting, clustering, or supervised learning.
- In browser apps, dimensionality reduction is often most valuable when it feeds directly into interactive visualizations.
