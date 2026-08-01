---
title: Robust Covariance in JavaScript with @kanaries/ml
description: Estimate robust covariance and detect multivariate outliers in JavaScript or TypeScript with MinCovDet and EllipticEnvelope from @kanaries/ml.
---

# Robust covariance in JavaScript

Robust covariance estimators describe the center and spread of multivariate data without letting a small number of extreme observations dominate the result. They are useful for anomaly detection, robust distance calculations, and preprocessing noisy tabular data.

`@kanaries/ml` provides browser- and Node.js-ready TypeScript implementations of FAST-MCD through [MinCovDet](minCovDet) and its anomaly-detection wrapper [EllipticEnvelope](ellipticEnvelope).

Choose `MinCovDet` when you need robust location, covariance, support masks, or Mahalanobis distances. Choose `EllipticEnvelope` when you need `predict`, `scoreSamples`, and `decisionFunction` with a contamination threshold.
