---
title: AdaBoostRegressor
description: API reference for AdaBoostRegressor
---

# Ensemble.AdaBoostRegressor

```ts
constructor(props?: { estimator?: DecisionTreeRegressor; n_estimators?: number; learning_rate?: number })
```

### Methods
+ `fit(trainX: number[][], trainY: number[]): void`
+ `predict(testX: number[][]): number[]`

### Example
```ts
const regr = new AdaBoostRegressor({ n_estimators: 100 });
regr.fit(X, y);
const pred = regr.predict([[0, 0, 0, 0]]);
```
