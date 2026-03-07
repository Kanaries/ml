---
title: LinearRegression
description: API and practical guide for LinearRegression in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: LinearRegression in JavaScript and TypeScript

LinearRegression models continuous targets with an interpretable linear relationship between features and outputs.

### When to use LinearRegression
- You need a transparent baseline for numeric prediction tasks.
- Feature-target relationships are approximately linear after transformation.
- You want fast training and low-latency inference in JavaScript.

### Implementation workflow
1. Prepare numeric features and split into train/validation sets.
2. Fit the model and inspect residual patterns for systematic errors.
3. Iterate on feature engineering when residuals show non-linear structure.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LinearRegression JavaScript`
- `LinearRegression TypeScript`
- `LinearRegression browser machine learning`
- `@kanaries/ml LinearRegression`

