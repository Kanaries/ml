---
title: SparsePCA
description: API reference for SparsePCA
---

# Decomposition.SparsePCA

Sparse Principal Components Analysis using truncated power iteration with soft thresholding. The algorithm stops when the updates change by less than `tol` or when `maxIter` is reached.

```ts
interface SparsePCAProps {
    nComponents?: number | null;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}
constructor(props: SparsePCAProps = {})
```

### Example
```ts
const transformer = new SparsePCA({ nComponents: 5, alpha: 0.1 });
transformer.fit(X);
const T = transformer.transform(X_test);
```
