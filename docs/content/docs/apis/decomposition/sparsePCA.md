---
title: SparsePCA
description: API reference for SparsePCA
---

# Decomposition.SparsePCA

Sparse Principal Components Analysis using truncated power iteration with soft thresholding. The algorithm stops when the updates change by less than `tol` or when `maxIter` is reached.

### Algorithm
Each component is extracted by iterative thresholding of the covariance matrix.
The process encourages sparsity by shrinking small coefficients towards zero.

```ts
interface SparsePCAProps {
    nComponents?: number | null;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}
constructor(props: SparsePCAProps = {})
```

### Parameters
- `nComponents` (number | null, default `null`): number of sparse components to compute. `null` keeps all components.
- `alpha` (number, default `1`): sparsity controlling parameter. Higher values lead to more zero coefficients.
- `maxIter` (number, default `100`): maximum number of iterations for each component.
- `tol` (number, default `1e-8`): stopping criterion for convergence of the iterative updates.

### Example
```ts
const transformer = new SparsePCA({ nComponents: 5, alpha: 0.1 });
transformer.fit(X);
const T = transformer.transform(X_test);
```
