---
title: LinearSVC
description: API reference for LinearSVC
---

# SVM.LinearSVC

```ts
interface LinearSVCProps {
    C?: number;
    maxIter?: number;
    learningRate?: number;
}
constructor(props: LinearSVCProps = {})
```

### Example
```ts
const svc = new LinearSVC();
svc.fit(X, y);
const result = svc.predict(T);
```
