---
title: Feature Selection in JavaScript with @kanaries/ml
description: Select informative columns in browser or Node.js machine-learning workflows with model-based, recursive, and univariate JavaScript feature selection in @kanaries/ml.
---

# Feature selection in JavaScript

Feature selection removes weak or redundant columns before prediction. It can reduce model cost, improve interpretability, and limit overfitting. `@kanaries/ml` provides model-based selectors and statistical scoring functions with a sklearn-like JavaScript API.

- [Model-based and recursive selectors](selectors): `SelectFromModel`, `RFE`, and `RFECV`
- [Univariate score functions](univariate): `chi2`, `fClassif`, `mutualInfoClassif`, and `mutualInfoRegression`
