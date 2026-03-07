---
title: AdaBoostRegressor
description: API and practical guide for AdaBoostRegressor in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: AdaBoostRegressor in JavaScript and TypeScript

AdaBoostRegressor combines weak regressors sequentially to improve predictive accuracy on non-linear regression patterns.

### When to use AdaBoostRegressor
- Single linear models underfit important non-linear behavior.
- You need stronger regression quality with manageable model complexity.
- Your deployment environment requires fast JS inference.

### Implementation workflow
1. Train with a conservative learning rate and enough estimators.
2. Validate MAE/RMSE on holdout data and monitor overfitting.
3. Tune estimator count and learning rate for stability vs accuracy.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `AdaBoostRegressor JavaScript`
- `AdaBoostRegressor TypeScript`
- `AdaBoostRegressor browser machine learning`
- `@kanaries/ml AdaBoostRegressor`

