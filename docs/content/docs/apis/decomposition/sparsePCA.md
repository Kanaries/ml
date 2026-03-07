---
title: SparsePCA
description: API and practical guide for SparsePCA in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
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

## Practical guide: SparsePCA in JavaScript and TypeScript

SparsePCA learns sparse components so each latent dimension uses only a subset of original features.

### When to use SparsePCA
- Interpretability of component-feature relationships is important.
- You need dimensionality reduction with built-in sparsity constraints.
- Dense PCA components are too hard to explain to stakeholders.

### Implementation workflow
1. Scale input features and choose sparsity-related hyperparameters.
2. Fit SparsePCA and inspect component loadings for feature selection signals.
3. Use sparse transformed outputs in linear or tree-based downstream models.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `SparsePCA JavaScript`
- `SparsePCA TypeScript`
- `SparsePCA browser machine learning`
- `@kanaries/ml SparsePCA`

