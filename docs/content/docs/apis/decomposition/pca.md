---
title: PCA
description: API reference for PCA
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
