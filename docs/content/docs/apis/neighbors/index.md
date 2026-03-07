---
title: Neighbors
description: Learn how to use Neighbors algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [KNearstNeighbors](knn)
- [BallTree](ballTree)
- [KDTree](kdTree)

## How to use the Neighbors module in real projects

The Neighbors module supports instance-based learning and efficient nearest-neighbor search for recommendations, retrieval, and similarity tasks.

### Selection checklist
1. Use KNearstNeighbors when you need a direct non-parametric classifier/regressor baseline.
2. Use KDTree or BallTree to speed up nearest-neighbor queries for repeated lookup workloads.
3. Tune distance metrics and neighborhood size based on recall/precision targets for your product.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `knn javascript`
- `kdtree typescript`
- `nearest neighbor search browser`

### Explore algorithms in this module
- [ballTree](ballTree)
- [kdTree](kdTree)
- [knn](knn)

