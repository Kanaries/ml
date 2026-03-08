---
title: Manifold Learning in JavaScript with @kanaries/ml
description: Explore t-SNE, MDS, Spectral Embedding, and Locally Linear Embedding in JavaScript and TypeScript with @kanaries/ml for visualization and neighborhood analysis.
---

# Manifold Learning in JavaScript

## Module overview

The Manifold module projects high-dimensional data into lower-dimensional embeddings while trying to preserve important geometric or neighborhood structure. These methods are primarily useful for visualization, exploratory analysis, and pre-processing workflows rather than for low-latency production inference.

This module is a strong fit when:

- you need 2D or 3D visualizations of high-dimensional data
- local neighborhood structure matters more than raw feature axes
- your embeddings will feed directly into browser charts or interactive analysis tools

## JavaScript implementation

`@kanaries/ml` provides manifold learning algorithms in JavaScript and TypeScript so teams can generate embeddings in the same runtime that renders charts, powers exploratory analysis, or prepares features in Node.js. This is especially practical for frontend-heavy tools where embedding output is consumed immediately by the UI.

If someone searches for "t-SNE in JavaScript" or "Spectral Embedding in TypeScript", this module is the right entry point.

## Quick navigation

- [t-SNE](tsne): local-neighborhood visualization for high-dimensional data
- [MDS](MDS): embeddings that preserve pairwise distance structure
- [Locally Linear Embedding](lle): non-linear manifold learning based on local reconstructions
- [Spectral Embedding](spectralEmbedding): graph-based low-dimensional embeddings

## Detailed module guide

### How to choose an algorithm

1. Use [t-SNE](tsne) for exploratory cluster visualization.
2. Use [MDS](MDS) when the main input or objective is a distance matrix.
3. Use [Locally Linear Embedding](lle) or [Spectral Embedding](spectralEmbedding) when local geometry or graph structure is central to the task.

### JavaScript deployment notes

- Treat these methods as exploration and preprocessing tools first, not as default request-time transforms.
- Run heavier embeddings outside the main browser thread when user interaction matters.
- Choose based on the structure you want to preserve: local neighborhoods, graph connectivity, or pairwise distances.
