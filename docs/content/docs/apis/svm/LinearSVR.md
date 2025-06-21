---
title: LinearSVR
description: API reference for LinearSVR
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
