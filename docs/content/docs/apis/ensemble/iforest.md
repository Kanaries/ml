---
title: IsolationForest
---

# Ensemble.IsolationForest

```ts
constructor(subsampling_size: number = 256, tree_num: number = 100, contamination: 'auto' | number = 'auto')
```

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
