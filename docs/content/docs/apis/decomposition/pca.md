---
title: PCA
description: API and practical guide for PCA in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Decomposition.PCA

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the SVD.

### Algorithm
The covariance matrix of the centered samples is decomposed by power iteration.
The resulting eigenvectors form the principal components sorted by explained variance.

```ts
constructor(nComponents: number | null = null)
```

### Parameters
- `nComponents` (number | null, default `null`): number of principal components to retain. If `null`, all components are used.

### Methods
- `fit(X: number[][]): void`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`
- `getComponents(): number[][]`
- `getMean(): number[]`
- `getExplainedVariance(): number[]`

### Example
```ts
const pca = new PCA(2);
pca.fit(X);
const T = pca.transform(X_test);
```

## Practical guide: PCA in JavaScript and TypeScript

PCA compresses correlated numeric features into orthogonal components while preserving maximal variance.

### When to use PCA
- Feature dimensions are high and training speed needs improvement.
- You need 2D/3D projections for exploratory visualization.
- Multicollinearity hurts stability in downstream supervised models.

### Implementation workflow
1. Standardize continuous features before decomposition.
2. Fit PCA and select component count by explained variance.
3. Use transformed features for visualization or downstream training.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `PCA JavaScript`
- `PCA TypeScript`
- `PCA browser machine learning`
- `@kanaries/ml PCA`

