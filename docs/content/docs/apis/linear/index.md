---
title: Linear Models in JavaScript with @kanaries/ml
description: Explore linear regression and logistic regression in JavaScript and TypeScript with @kanaries/ml for interpretable supervised learning.
---

# Linear Models in JavaScript

## Module overview

The Linear module provides simple, fast, and interpretable supervised learning algorithms for common tabular problems. These models are often the best starting point when you want a strong baseline before moving to trees, SVMs, or ensembles.

This module is a strong fit when:

- you need transparent coefficients and easy-to-debug predictions
- training speed and low implementation overhead matter
- you are solving common regression or binary classification tasks

## JavaScript implementation

`@kanaries/ml` provides core linear models in JavaScript and TypeScript so teams can keep prediction logic close to application code. This is especially useful for browser demos, pricing tools, forecasting helpers, and product features where stakeholders care about both the result and the reasoning behind it.

If someone searches for "linear regression in JavaScript" or "logistic regression in TypeScript", this module is the right entry point.

## Quick navigation

- [Linear Regression](linearRegression): numeric prediction with interpretable coefficients
- [Logistic Regression](logisticRegression): binary classification with probability-oriented outputs

## Detailed module guide

### How to choose an algorithm

1. Use [Linear Regression](linearRegression) for continuous targets.
2. Use [Logistic Regression](logisticRegression) for binary outcomes and threshold-based decisions.
3. Start here before trying more complex non-linear models so you have a clear baseline.

### JavaScript deployment notes

- Normalize numeric inputs when optimization stability matters.
- Linear models are often ideal when you need predictions to remain explainable to both engineers and stakeholders.
- Keep them in the application layer when the product benefits from transparent, low-latency inference.
