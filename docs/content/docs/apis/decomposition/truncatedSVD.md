---
title: TruncatedSVD
description: API reference for TruncatedSVD
---

# Decomposition.TruncatedSVD

Dimensionality reduction using truncated Singular Value Decomposition of the data. Unlike PCA, the input data is not centered before decomposition.

```ts
constructor(nComponents: number = 2)
```

### Methods
- `fit(X: number[][]): void`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `inverseTransform(X: number[][]): number[][]`
- `getComponents(): number[][]`
- `getSingularValues(): number[]`
- `getExplainedVariance(): number[]`
- `getExplainedVarianceRatio(): number[]`

### Example
```ts
const svd = new TruncatedSVD(2);
svd.fit(X);
const T = svd.transform(X_test);
```
