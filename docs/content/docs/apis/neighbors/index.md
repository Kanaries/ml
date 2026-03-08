---
title: Nearest-Neighbor Algorithms in JavaScript with @kanaries/ml
description: Explore k-nearest neighbors, Ball Tree, and KD Tree in JavaScript and TypeScript with @kanaries/ml for similarity search, retrieval, and local prediction.
---

# Nearest-Neighbor Algorithms in JavaScript

## Module overview

The Neighbors module supports instance-based learning and efficient nearest-neighbor search. It is useful for similarity tasks, recommendation flows, retrieval systems, and classifiers that reason directly from nearby examples instead of learning a compact parametric model.

This module is a strong fit when:

- prediction or ranking depends on nearby examples
- you need repeated nearest-neighbor queries over a relatively stable dataset
- you want to compare a simple KNN baseline with explicit search structures

## JavaScript implementation

`@kanaries/ml` provides nearest-neighbor algorithms in JavaScript and TypeScript so teams can keep indexing, lookup, and local prediction inside the same browser or Node.js runtime as the rest of the product logic. This is especially useful for recommendation tools, search experiences, and interactive similarity workflows that need low-friction integration with application code.

If someone searches for "k-nearest neighbors in JavaScript", "KD Tree in JavaScript", or "Ball Tree in TypeScript", this module is the right entry point.

## Quick navigation

- [k-Nearest Neighbors](knn): direct prediction from nearby labeled examples
- [Ball Tree](ballTree): faster repeated nearest-neighbor search in metric spaces
- [KD Tree](kdTree): efficient spatial indexing for repeated lookup on suitable data

## Detailed module guide

### How to choose an algorithm

1. Use [k-Nearest Neighbors](knn) when you want a direct non-parametric baseline.
2. Use [Ball Tree](ballTree) or [KD Tree](kdTree) when repeated query speed matters.
3. Benchmark both the quality impact of distance choices and the latency impact of your search structure.

### JavaScript deployment notes

- Scale numeric features before distance-based prediction or lookup.
- Build tree-based indexes once and reuse them when the reference dataset changes slowly.
- Keep nearest-neighbor search close to the application layer when retrieval results immediately drive UI or product logic.
