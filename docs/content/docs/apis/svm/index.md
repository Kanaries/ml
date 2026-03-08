---
title: Support Vector Machines in JavaScript with @kanaries/ml
description: Explore SVC, NuSVC, LinearSVC, and LinearSVR in JavaScript and TypeScript with @kanaries/ml for margin-based classification and regression.
---

# Support Vector Machines in JavaScript

## Module overview

The SVM module provides margin-based models for classification and regression. These algorithms are useful when you want strong decision boundaries, controlled regularization, and model families that range from simple linear separators to kernelized non-linear classifiers.

This module is a strong fit when:

- linear models are too weak or margin-based behavior is preferred
- you need both linear and kernel-based options in one family
- you want to compare classification and regression variants with similar modeling ideas

## JavaScript implementation

`@kanaries/ml` provides support vector machine models in JavaScript and TypeScript so teams can run margin-based learning in the same runtime that already owns feature extraction, request handling, and product logic. This is useful in both browser tools and Node.js services when Python is not the serving environment.

If someone searches for "SVM in JavaScript", "LinearSVC in TypeScript", or "SVC in JavaScript", this module is the right entry point.

## Quick navigation

- [SVC](SVC): kernel classification for non-linear decision boundaries
- [NuSVC](NuSVC): support vector classification with `nu`-based control
- [LinearSVC](LinearSVC): linear classification for suitable feature spaces
- [LinearSVR](LinearSVR): linear support vector regression

## Detailed module guide

### How to choose an algorithm

1. Use [SVC](SVC) or [NuSVC](NuSVC) when non-linear classification behavior matters.
2. Use [LinearSVC](LinearSVC) when a linear boundary is sufficient and feature spaces are larger or simpler.
3. Use [LinearSVR](LinearSVR) when you want a support-vector-style regression objective instead of plain least squares.

### JavaScript deployment notes

- Scale features before training because SVM behavior is highly sensitive to feature magnitude.
- Treat kernel choice and regularization settings as part of the same tuning problem.
- Keep these models in JS when their output needs to connect directly to application-side business logic.
