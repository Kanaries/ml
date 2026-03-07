---
title: HDBScan
description: API and practical guide for HDBScan in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Clusters.HDBScan

```ts
constructor(
    min_cluster_size: number = 5,
    min_samples: number | null = null,
    cluster_selection_epsilon: number = 0.5,
    metric: Distance.IDistanceType = 'euclidiean'
)
```

`fitPredict(samplesX: number[][]): number[]` returns cluster labels. Noise points are marked as `-1`.

This is a simplified implementation that internally calls DBSCAN using `cluster_selection_epsilon` as the `eps` parameter.

```ts
const hdb = new HDBScan(5, null, 0.6);
const labels = hdb.fitPredict(X);
```

## Practical guide: HDBScan in JavaScript and TypeScript

HDBScan finds clusters with varying density and identifies noise points without forcing every point into a cluster.

### When to use HDBScan
- Cluster shapes are irregular and density changes across regions.
- Outlier/noise detection is important for your downstream workflow.
- You do not know an exact cluster count in advance.

### Implementation workflow
1. Scale features so distance comparisons are meaningful.
2. Fit the model and inspect cluster labels plus noise assignments.
3. Use cluster stability and business metrics to tune density settings.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `HDBScan JavaScript`
- `HDBScan TypeScript`
- `HDBScan browser machine learning`
- `@kanaries/ml HDBScan`

