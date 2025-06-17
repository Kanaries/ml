---
title: LinearSVC
description: API reference for LinearSVC
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
