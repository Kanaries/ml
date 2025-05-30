---
title: MeanShift
description: API reference for MeanShift
---

# Clusters.MeanShift

```ts
constructor(
    bandwidth: number = 1,
    max_iter: number = 300,
    distanceType: Distance.IDistanceType = 'euclidiean'
)
```

Methods:
- `fitPredict(samplesX: number[][]): number[]`
- `getCentroids(): number[][]`

```ts
const ms = new MeanShift(2);
const labels = ms.fitPredict(X);
const centers = ms.getCentroids();
```
