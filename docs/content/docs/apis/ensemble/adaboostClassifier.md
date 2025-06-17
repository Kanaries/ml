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
