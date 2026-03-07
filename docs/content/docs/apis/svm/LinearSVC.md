---
title: LinearSVC
description: API and practical guide for LinearSVC in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SVM.LinearSVC

Linear support vector classifier optimized with gradient descent on hinge loss. It trains one-versus-rest linear models to separate classes.

```ts
interface LinearSVCProps {
    C?: number;
    maxIter?: number;
    learningRate?: number;
}
constructor(props: LinearSVCProps = {})
```

### Parameters
- `C` (number, default `1`): regularization strength
- `maxIter` (number, default `100`): maximum training iterations
- `learningRate` (number, default `0.01`): step size for gradient descent

### Example
```ts
const svc = new LinearSVC();
svc.fit(X, y);
const result = svc.predict(T);
```

## Practical guide: LinearSVC in JavaScript and TypeScript

LinearSVC trains linear support vector classifiers optimized for high-dimensional and sparse feature spaces.

### When to use LinearSVC
- You have many features (for example text vectors) and need fast linear classification.
- Margin-based classification outperforms logistic baselines in your tests.
- You need robust performance with controlled model complexity.

### Implementation workflow
1. Scale/normalize features and choose regularization strength.
2. Fit the classifier and evaluate margin-driven classification metrics.
3. Tune `C` and class weighting to match recall/precision priorities.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LinearSVC JavaScript`
- `LinearSVC TypeScript`
- `LinearSVC browser machine learning`
- `@kanaries/ml LinearSVC`

