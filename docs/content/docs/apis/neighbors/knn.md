---
title: KNearstNeighbors
description: API and practical guide for KNearstNeighbors in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Neighbors.KNearstNeighbors

```ts
constructor(
    kNeighbors: number = 5,
    weightType: IWeightType = 'uniform',
    distanceType: Distance.IDistanceType = 'euclidiean',
    pNorm: number = 2
)
```

### Parameters

- `kNeighbors` *(number)*: number of neighbors used for prediction. Default is `5`.
- `weightType` *(`'uniform' | 'distance'`)*: weighting strategy for voting. `'uniform'`
  counts every neighbor equally while `'distance'` weighs closer samples more.
- `distanceType` *(Distance.IDistanceType)*: distance metric. Defaults to
  `'euclidiean'` but other metrics from `Distance` can be used.
- `pNorm` *(number)*: order of the norm when using Minkowski distance. Default is
  `2`.

### Algorithm

KNN is a lazy classifier. During prediction it computes the distance between the
query sample and all training points. The closest `kNeighbors` points vote for
the label. Voting can be uniform or weighted by inverse distance depending on
`weightType`.

### Example
```ts
const knn = new KNearstNeighbors(3, 'distance', '2-norm');
knn.fit(trainX, trainY);
const result = knn.predict(trainX);
```

## Practical guide: KNearstNeighbors in JavaScript and TypeScript

KNearstNeighbors predicts from nearby examples and works as a strong non-parametric baseline for classification and regression.

### When to use KNearstNeighbors
- Decision boundaries are irregular and hard to model parametrically.
- You need straightforward behavior that is easy to reason about.
- Training time should be minimal and inference latency is acceptable.

### Implementation workflow
1. Scale numeric features and choose a distance metric.
2. Fit on labeled examples and test multiple `k` values.
3. Validate accuracy/latency tradeoffs before deployment.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `KNearstNeighbors JavaScript`
- `KNearstNeighbors TypeScript`
- `KNearstNeighbors browser machine learning`
- `@kanaries/ml KNearstNeighbors`

