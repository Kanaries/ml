---
title: LinearRegression
description: API reference for LinearRegression
---

# Linear.LinearRegression

```ts
constructor()
```

This class implements ordinary least squares linear regression. It estimates
coefficients for a linear model by minimizing the squared error between
predicted and actual values.

### Methods
+ `fit(X: number[][], Y: number[]): void`
+ `predict(X: number[][]): number[]`

#### fit
* `X` - Feature matrix of shape `[nSamples, nFeatures]`.
* `Y` - Target values of length `nSamples`.

#### predict
* `X` - Feature matrix for which to compute predictions.

### Example
```ts
const lr = new LinearRegression();
lr.fit([[0], [1]], [1, 3]);
const pred = lr.predict([[2]]); // about 5
```
