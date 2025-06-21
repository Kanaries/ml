---
title: LabelPropagation
description: API reference for LabelPropagation
---

# SemiSupervised.LabelPropagation

```ts
interface LabelPropagationOptions {
    kernel?: 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);
    gamma?: number;
    nNeighbors?: number;
    maxIter?: number;
    tol?: number;
}
constructor(options: LabelPropagationOptions = {})
```

Label propagation assigns labels to unlabeled data by propagating
information from labeled points across a graph defined by a kernel.

### Methods
* `fit(trainX: number[][], trainY: number[]): void`
* `predict(testX: number[][]): number[]`
* `predictProba(testX: number[][]): number[][]`

#### Example
```ts
const lp = new LabelPropagation();
lp.fit(trainX, trainY);
const preds = lp.predict(testX);
```
