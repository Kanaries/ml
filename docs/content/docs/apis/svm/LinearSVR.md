---
title: LinearSVR
description: API and practical guide for LinearSVR in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SVM.LinearSVR

Linear support vector regression trained with gradient descent on the epsilon-insensitive loss.

```ts
interface LinearSVRProps {
    epsilon?: number;
    C?: number;
    maxIter?: number;
    learningRate?: number;
}
constructor(props: LinearSVRProps = {})
```

### Parameters
- `epsilon` (number, default `0`): width of the insensitive tube around the regression line
- `C` (number, default `1`): regularization strength
- `maxIter` (number, default `100`): maximum training iterations
- `learningRate` (number, default `0.01`): optimizer step size

### Example
```ts
const reg = new LinearSVR();
reg.fit(X, y);
const preds = reg.predict(T);
```

## Practical guide: LinearSVR in JavaScript and TypeScript

LinearSVR performs margin-based linear regression with robustness to moderate outliers in target values.

### When to use LinearSVR
- You need regression in high-dimensional spaces with linear assumptions.
- Outlier sensitivity from ordinary least squares is causing instability.
- You want a fast margin-based baseline in Node.js services.

### Implementation workflow
1. Prepare standardized numeric features and choose epsilon margin.
2. Fit LinearSVR and measure MAE/RMSE on holdout data.
3. Tune `C` and epsilon for error tolerance versus fit quality.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LinearSVR JavaScript`
- `LinearSVR TypeScript`
- `LinearSVR browser machine learning`
- `@kanaries/ml LinearSVR`

