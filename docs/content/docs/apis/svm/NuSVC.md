---
title: NuSVC
description: API reference for NuSVC
---

# SVM.NuSVC

Variant of `SVC` that uses the `nu` parameter to control the fraction of support vectors and the margin errors.

```ts
interface NuSVCProps {
    nu?: number;
}
constructor(props: NuSVCProps = {})
```

### Parameters
- `nu` (number, default `0.5`): trade-off between the number of support vectors and training errors
- `C` (number, default `1/nu`): regularization strength
- `maxIter` (number, default `100`): maximum iterations for training
- `learningRate` (number, default `0.01`): step size for gradient descent
- `kernel` ('linear' | 'rbf', default `'rbf'`): kernel type
- `gamma` (number, default `1`): RBF kernel coefficient

NuSVC shares the same usage as `SVC` but uses the `nu` parameter instead of `C`.
