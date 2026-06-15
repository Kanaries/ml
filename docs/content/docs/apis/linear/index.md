---
title: Linear Models in JavaScript with @kanaries/ml
description: Explore linear regression, logistic regression, regularized regression, and linear classification in JavaScript and TypeScript with @kanaries/ml.
---

# Linear Models in JavaScript

## Module overview

The Linear module provides simple, fast, and interpretable supervised learning algorithms for common tabular problems. These models are often the best starting point when you want a strong baseline before moving to trees, SVMs, or ensembles.

This module is a strong fit when:

- you need transparent coefficients and easy-to-debug predictions
- training speed and low implementation overhead matter
- you are solving common regression, regularized regression, or classification tasks

## JavaScript implementation

`@kanaries/ml` provides core linear models in JavaScript and TypeScript so teams can keep prediction logic close to application code. This is especially useful for browser demos, pricing tools, forecasting helpers, and product features where stakeholders care about both the result and the reasoning behind it.

If someone searches for "linear regression in JavaScript" or "logistic regression in TypeScript", this module is the right entry point.

## Quick navigation

- [Linear Regression](linearRegression): numeric prediction with interpretable coefficients
- [Polynomial Regression](polynomialRegression): curve fitting through polynomial feature expansion
- [Ridge Regression](ridgeRegression): L2-regularized numeric prediction
- [Lasso Regression](lassoRegression): L1-regularized numeric prediction with sparse coefficients
- [Ridge](ridge): sklearn-style alias for RidgeRegression
- [Lasso](lasso): sklearn-style alias for LassoRegression
- [ElasticNet](elasticNet): combined L1 and L2 regularized regression
- [RidgeClassifier](ridgeClassifier): one-vs-rest L2-regularized linear classification
- [Logistic Regression](logisticRegression): binary classification with probability-oriented outputs

## Detailed module guide

### How to choose an algorithm

1. Use [Linear Regression](linearRegression) for continuous targets.
2. Use [Ridge Regression](ridgeRegression), [Lasso Regression](lassoRegression), or [ElasticNet](elasticNet) when plain linear regression overfits or coefficients are unstable.
3. Use [Polynomial Regression](polynomialRegression) when a smooth nonlinear relationship is enough.
4. Use [Logistic Regression](logisticRegression) or [RidgeClassifier](ridgeClassifier) for classification.

### JavaScript deployment notes

- Normalize numeric inputs when optimization stability matters.
- Linear models are often ideal when you need predictions to remain explainable to both engineers and stakeholders.
- Keep them in the application layer when the product benefits from transparent, low-latency inference.
