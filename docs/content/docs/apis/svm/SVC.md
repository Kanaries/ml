---
title: SVC
---

# SVM.SVC

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

### Example
```ts
const svc = new SVC({ kernel: 'linear' });
svc.fit(X, y);
const result = svc.predict(T);
```
