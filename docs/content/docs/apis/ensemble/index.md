---
title: Ensemble Learning in JavaScript with @kanaries/ml
description: Explore ensemble learning algorithms in JavaScript and TypeScript with @kanaries/ml, including Isolation Forest and AdaBoost models for anomaly detection, classification, and regression.
---

# Ensemble Learning in JavaScript

## Module overview

The Ensemble module combines multiple weak learners to create stronger models for anomaly detection, classification, and regression. These algorithms are useful when single-model baselines are close to useful but need more robustness or more expressive decision behavior.

This module is a strong fit when:

- you need anomaly detection on tabular product or telemetry data
- a simple classifier or regressor underfits important patterns
- you want stronger performance without moving all the way to heavier model families

## JavaScript implementation

`@kanaries/ml` provides ensemble models in JavaScript and TypeScript so teams can run boosted learners and anomaly detection inside browser products and Node.js services. This is particularly helpful when feature generation, threshold logic, and downstream product actions already live in JS and should stay close to the model runtime.

If someone searches for "Isolation Forest in JavaScript" or "AdaBoost in TypeScript", this module is the right entry point.

## Quick navigation

- [Isolation Forest](iforest): anomaly detection for mostly-normal datasets with rare outliers
- [AdaBoost Regressor](adaboost): boosted regression for non-linear tabular prediction
- [AdaBoost Classifier](adaboostClassifier): boosted classification for harder decision boundaries

## Detailed module guide

### How to choose an algorithm

1. Use [Isolation Forest](iforest) when labels are unavailable and the main goal is outlier detection.
2. Use [AdaBoost Classifier](adaboostClassifier) when a simple classifier misses hard examples.
3. Use [AdaBoost Regressor](adaboost) when a single regressor is too weak for the target pattern.

### JavaScript deployment notes

- Treat thresholds, estimator counts, and learning rates as product-level tuning decisions.
- Ensemble models often provide a good middle ground between simple baselines and more operationally complex systems.
- In JS applications, keep feature extraction and prediction close together so model output can flow directly into business logic.
