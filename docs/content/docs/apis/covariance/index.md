---
title: Robust Covariance in JavaScript with @kanaries/ml
description: Estimate empirical, shrinkage, sparse Graphical Lasso, and robust FAST-MCD covariance matrices in JavaScript and TypeScript with @kanaries/ml.
---

## Module overview

A **covariance matrix** captures the pairwise variances and correlations among numeric features. Accurate covariance (and its inverse, the **precision matrix**) powers Mahalanobis distance calculations, multivariate outlier detection, Gaussian scoring, portfolio risk modeling, and conditional-independence graph discovery.

However, the sample covariance matrix breaks down in two common situations:

1. **Outliers contaminate the cloud**: a handful of corrupted rows can inflate variances and rotate eigenvectors away from the true inlier distribution—solved by robust estimators (`MinCovDet`, `EllipticEnvelope`).
2. **High dimensionality relative to sample size (`p` near or exceeding `n`)**: empirical covariance becomes ill-conditioned or singular—solved by shrinkage (`ShrunkCovariance`, `LedoitWolf`, `OAS`) or sparse inverse covariance (`GraphicalLasso`).

## JavaScript implementation

`@kanaries/ml` provides the complete scikit-learn `covariance` estimator family under `import { Covariance } from '@kanaries/ml'` in pure TypeScript for browser and Node.js runtimes:

| Estimator | Method | Best for |
| --- | --- | --- |
| [`Covariance.EmpiricalCovariance`](/docs/apis/covariance/covarianceEstimators) | Maximum-likelihood covariance | Clean datasets where `n >> p` |
| [`Covariance.ShrunkCovariance`](/docs/apis/covariance/covarianceEstimators) | Fixed shrinkage toward scaled identity | Manual regularization of noisy covariance |
| [`Covariance.LedoitWolf`](/docs/apis/covariance/covarianceEstimators) | Analytical MSE-optimal shrinkage | General-purpose well-conditioned covariance |
| [`Covariance.OAS`](/docs/apis/covariance/covarianceEstimators) | Oracle Approximating Shrinkage | Gaussian-like high-dimensional datasets |
| [`Covariance.GraphicalLasso`](/docs/apis/covariance/covarianceEstimators) | L1-penalized sparse precision matrix | Discovering sparse conditional independence graphs |
| [`Covariance.MinCovDet`](/docs/apis/covariance/minCovDet) | FAST-MCD robust location & scatter | Robust Mahalanobis distances with outlier resistance |
| [`Covariance.EllipticEnvelope`](/docs/apis/covariance/ellipticEnvelope) | Robust Gaussian anomaly detector | Flagging multivariate outliers (`1` inlier, `-1` outlier) |

## Quick start example

```ts
import { Covariance } from '@kanaries/ml';

const X = [[0, 0], [0.1, 0.2], [0.2, 0.1], [-0.1, 0], [8, 8]];
const detector = new Covariance.EllipticEnvelope({ contamination: 0.2, randomState: 42 });
detector.fit(X);
console.log(detector.predict(X)); // [1, 1, 1, 1, -1]
```

## Choosing the right covariance estimator

- Need **multivariate outlier detection** on roughly elliptical data? Start with [`EllipticEnvelope`](/docs/apis/covariance/ellipticEnvelope) (or compare with [`IsolationForest`](/docs/apis/ensemble/iforest) and [`LocalOutlierFactor`](/docs/apis/neighbors/localOutlierFactor) in our [outlier methods guide](/docs/guides/outlier-methods)).
- Need **robust location, scatter, or Mahalanobis distances** without hard thresholding? Use [`MinCovDet`](/docs/apis/covariance/minCovDet).
- Need **well-conditioned covariance or sparse precision** for downstream modeling? Use [`LedoitWolf`, `OAS`, or `GraphicalLasso`](/docs/apis/covariance/covarianceEstimators).
