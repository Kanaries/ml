---
title: AdaBoostClassifier
description: API reference for AdaBoostClassifier
---

# Ensemble.AdaBoostClassifier

```ts
interface AdaBoostClassifierProps {
    nEstimators?: number;
    learningRate?: number;
    randomState?: number;
}
constructor(props: AdaBoostClassifierProps = {})
```

### Parameters
| name | type | default | description |
|-|-|-|-|
| nEstimators | number | 50 | Number of boosting iterations |
| learningRate | number | 1.0 | Weight applied to each stump |
| randomState | number | undefined | Seed for reproducibility |

### Algorithm
AdaBoostClassifier trains decision stumps sequentially and reweights samples so that misclassified points receive more focus in subsequent rounds.

### Methods

+ `fit(trainX: number[][], trainY: number[]): void`
+ `predict(testX: number[][]): number[]`
+ `predictProba(testX: number[][]): number[][]`
+ `getFeatureImportances(): number[]`

### Example
```ts
const clf = new AdaBoostClassifier();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```
