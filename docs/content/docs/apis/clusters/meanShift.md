---
title: MeanShift
description: API and practical guide for MeanShift in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: MeanShift in JavaScript and TypeScript

MeanShift detects high-density regions and infers cluster count automatically from data density.

### When to use MeanShift
- You want clustering without specifying the number of clusters.
- Density peaks are more meaningful than centroid partitions.
- Your product needs adaptive grouping behavior over time.

### Implementation workflow
1. Scale numeric features and choose bandwidth heuristics.
2. Fit and inspect discovered modes and assigned labels.
3. Tune bandwidth to balance over-fragmentation and over-merging.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `MeanShift JavaScript`
- `MeanShift TypeScript`
- `MeanShift browser machine learning`
- `@kanaries/ml MeanShift`

