---
title: HDBScan
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
