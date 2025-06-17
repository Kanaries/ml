---
title: AdaBoostRegressor
description: API reference for AdaBoostRegressor
---

# Ensemble.AdaBoostRegressor

```ts
constructor(props?: { estimator?: DecisionTreeRegressor; n_estimators?: number; learning_rate?: number })
```

### Parameters
| name | type | default | description |
|-|-|-|-|
| estimator | DecisionTreeRegressor | depth 3 tree | Base learner used in boosting |
| n_estimators | number | 50 | Number of boosting rounds |
| learning_rate | number | 1.0 | Shrinks the contribution of each regressor |

### Algorithm
AdaBoostRegressor combines weak regressors sequentially. Each new estimator focuses on samples that previous models predicted poorly.

### Methods
+ `fit(trainX: number[][], trainY: number[]): void`
+ `predict(testX: number[][]): number[]`

### Example
```ts
const regr = new AdaBoostRegressor({ n_estimators: 100 });
regr.fit(X, y);
const pred = regr.predict([[0, 0, 0, 0]]);
```
