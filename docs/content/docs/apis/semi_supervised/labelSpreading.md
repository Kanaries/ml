---
title: LabelSpreading
description: API reference for LabelSpreading
---

# SemiSupervised.LabelSpreading

```ts
interface LabelSpreadingOptions {
    kernel?: 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);
    gamma?: number;
    nNeighbors?: number;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}
constructor(options: LabelSpreadingOptions = {})
```

Label spreading assigns labels to unlabeled data using a normalized graph
and soft clamping controlled by `alpha`.

### Methods
* `fit(trainX: number[][], trainY: number[]): void`
* `predict(testX: number[][]): number[]`
* `predictProba(testX: number[][]): number[][]`

#### Example
```ts
const ls = new LabelSpreading();
ls.fit(trainX, trainY);
const preds = ls.predict(testX);
```
