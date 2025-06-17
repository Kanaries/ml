---
title: KNearstNeighbors
description: API reference for KNearstNeighbors
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
