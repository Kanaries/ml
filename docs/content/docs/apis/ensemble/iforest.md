---
title: IsolationForest
description: API reference for IsolationForest
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
