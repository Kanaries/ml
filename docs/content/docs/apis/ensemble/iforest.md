---
title: IsolationForest
description: API and practical guide for IsolationForest in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Ensemble.IsolationForest

```ts
constructor(subsampling_size: number = 256, tree_num: number = 100, contamination: 'auto' | number = 'auto')
```

### Parameters
| name | type | default | description |
|-|-|-|-|
| subsampling_size | number | 256 | Number of samples used to build each tree |
| tree_num | number | 100 | Number of isolation trees in the forest |
| contamination | 'auto' \| number | 'auto' | Expected proportion of outliers |

### Algorithm
IsolationForest randomly splits features to isolate samples. Points that can be isolated with fewer splits are considered anomalies.

### Methods
+ `fit(samplesX: number[][]): void`
+ `predict(samplesX: number[][]): (0|1)[]`

### Example
```ts
const iForest = new IsolationForest(256, 10, 0.25);
const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1]];
iForest.fit(X);
const result = iForest.predict(X);
```

## Practical guide: IsolationForest in JavaScript and TypeScript

IsolationForest detects anomalies by isolating rare points with shorter path lengths in random partition trees.

### When to use IsolationForest
- You have mostly normal behavior with relatively few outliers.
- Labeling anomalies is expensive or unavailable.
- You need near-real-time anomaly scoring in browser or Node.js.

### Implementation workflow
1. Train on representative mostly-normal historical samples.
2. Predict anomaly labels or scores on incoming events.
3. Tune contamination and decision thresholds using alert precision targets.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `IsolationForest JavaScript`
- `IsolationForest TypeScript`
- `IsolationForest browser machine learning`
- `@kanaries/ml IsolationForest`

