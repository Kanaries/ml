---
title: SVC
description: API reference for SVC
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
