---
title: SVC
description: API and practical guide for SVC in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SVM.SVC

Support Vector Classification. When the kernel is set to `rbf` it applies the kernel trick to transform input features before training a linear classifier.

```ts
interface SVCProps {
    C?: number;
    maxIter?: number;
    learningRate?: number;
    kernel?: 'linear' | 'rbf';
    gamma?: number;
}
constructor(props: SVCProps = {})
```

### Parameters
- `C` (number, default `1`): regularization strength
- `maxIter` (number, default `100`): maximum iterations for training
- `learningRate` (number, default `0.01`): step size for the optimizer
- `kernel` ('linear' | 'rbf', default `'rbf'`): kernel type
- `gamma` (number, default `1`): RBF kernel coefficient

### Example
```ts
const svc = new SVC({ kernel: 'linear' });
svc.fit(X, y);
const result = svc.predict(T);
```

## Practical guide: SVC in JavaScript and TypeScript

SVC trains kernelized support vector classifiers for non-linear decision boundaries.

### When to use SVC
- Linear models underfit complex class separation patterns.
- You can afford kernel-based training for improved boundary flexibility.
- You need strong classification performance on medium-sized datasets.

### Implementation workflow
1. Scale features and pick a kernel (RBF is a common starting point).
2. Fit with baseline hyperparameters and assess validation metrics.
3. Tune `C`, kernel settings, and class weights for task-specific goals.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `SVC JavaScript`
- `SVC TypeScript`
- `SVC browser machine learning`
- `@kanaries/ml SVC`

