---
title: OPTICS
description: API and practical guide for OPTICS in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: OPTICS in JavaScript and TypeScript

OPTICS captures density-based cluster structure across multiple scales and handles varying-density datasets.

### When to use OPTICS
- DBSCAN-style sensitivity to a single epsilon is too limiting.
- You need ordering information to inspect hierarchical density structure.
- Noise handling is required for reliable segmentation.

### Implementation workflow
1. Prepare distance-scaled features and configure neighborhood constraints.
2. Fit and analyze reachability or extracted cluster labels.
3. Tune min-samples and extraction thresholds for your target behavior.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `OPTICS JavaScript`
- `OPTICS TypeScript`
- `OPTICS browser machine learning`
- `@kanaries/ml OPTICS`

