---
title: TruncatedSVD
description: API and practical guide for TruncatedSVD in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Decomposition.TruncatedSVD

Dimensionality reduction using truncated Singular Value Decomposition of the data. Unlike PCA, the input data is not centered before decomposition.

### Algorithm
The algorithm performs power iteration on the uncentered covariance matrix and keeps the top components corresponding to the largest singular values.

```ts
constructor(nComponents: number = 2)
```

### Parameters
- `nComponents` (number, default `2`): number of singular vectors to retain.

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

## Practical guide: TruncatedSVD in JavaScript and TypeScript

TruncatedSVD reduces dimensionality of large sparse matrices without centering, making it practical for text/vectorized data.

### When to use TruncatedSVD
- Your feature matrix is sparse (for example TF-IDF or one-hot encodings).
- You need latent semantic compression before classification or retrieval.
- PCA centering is too expensive or undesirable for sparse data.

### Implementation workflow
1. Build sparse-compatible numeric input matrix.
2. Fit TruncatedSVD with candidate component counts.
3. Evaluate retrieval/classification quality on reduced features.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `TruncatedSVD JavaScript`
- `TruncatedSVD TypeScript`
- `TruncatedSVD browser machine learning`
- `@kanaries/ml TruncatedSVD`

