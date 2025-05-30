---
title: OPTICS
description: API reference for OPTICS
---

# Clusters.OPTICS

```ts
interface OPTICSOptions {
    min_samples?: number;
    max_eps?: number;
    metric?: Distance.IDistanceType;
    p?: number;
    eps?: number;
}
constructor(options: OPTICSOptions = {})
```

`fitPredict(samplesX: number[][]): number[]` returns cluster labels. Noise points are marked as `-1`.

```ts
const optics = new OPTICS({ eps: 0.5, min_samples: 5 });
const labels = optics.fitPredict(X);
```
