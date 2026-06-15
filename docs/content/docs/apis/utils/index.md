---
title: Machine Learning Utilities in JavaScript with @kanaries/ml
description: Explore machine learning workflow utilities in JavaScript and TypeScript with @kanaries/ml, including preprocessing, sampling, model selection, statistics, and async execution helpers.
---

# Machine Learning Utilities in JavaScript

## Module overview

The Utils module contains workflow helpers that make machine learning code easier to integrate into real JavaScript applications. These utilities are useful when the challenge is preparing data, validating models, or making model execution behave well inside browser or Node.js environments.

This module is a strong fit when:

- you need preprocessing, sampling, model selection, or simple statistics
- you need non-blocking execution around training or inference
- application responsiveness matters as much as model quality
- you want ML helpers that work naturally with the rest of a JS stack

## JavaScript implementation

`@kanaries/ml` provides utility helpers in JavaScript and TypeScript so teams can shape data preparation, validation, execution behavior, integration patterns, and developer ergonomics without leaving the main application runtime. These helpers are especially useful in UI-heavy products and services where ML work should not block interaction or event-loop responsiveness.

If someone searches for "machine learning utilities in JavaScript", "preprocessing in TypeScript", or "async ML execution in TypeScript", this module is the right entry point.

## Quick navigation

- [Preprocessing](preprocessing): scale, normalize, encode, impute, and select features
- [Sampling](sampling): sample arrays and create train/test splits
- [ModelSelection](modelSelection): run K-fold splits, cross-validation, grid search, and randomized search
- [Stat](stat): compute lightweight statistics used in ML workflows
- [asyncMode](asyncMode): run synchronous functions in a worker-like async execution path

## Detailed module guide

### How to use this module

1. Start with [Preprocessing](preprocessing) and [Sampling](sampling) to make training data model-ready.
2. Use [ModelSelection](modelSelection) to compare estimators and tune parameters.
3. Use [asyncMode](asyncMode) when training or inference should not block the main browser thread or Node.js event loop.
4. Pair utilities with model modules so application behavior stays predictable under load.

### JavaScript deployment notes

- Utilities matter most when ML code runs inside interactive or latency-sensitive product surfaces.
- Use them to keep heavy work off the main thread when responsiveness is part of the user experience.
- Combine them with the rest of the API catalog to build production-friendly JS ML workflows.
