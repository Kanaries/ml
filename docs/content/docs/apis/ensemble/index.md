---
title: Ensemble Learning in JavaScript with @kanaries/ml
description: Explore ensemble learning algorithms in JavaScript and TypeScript with @kanaries/ml, including Isolation Forest, AdaBoost, random forests, and bagging models.
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
- [RandomForestClassifier](randomForestClassifier): majority-vote tree ensembles for classification
- [RandomForestRegressor](randomForestRegressor): averaged tree ensembles for regression
- [BaggingClassifier](baggingClassifier): bootstrap classifier ensembles with default or custom estimators
- [AdaBoost Regressor](adaboost): boosted regression for non-linear tabular prediction
- [AdaBoost Classifier](adaboostClassifier): boosted classification for harder decision boundaries
- [GradientBoostingRegressor](gradientBoostingRegressor): squared-error gradient boosting for tabular regression
- [GradientBoostingClassifier](gradientBoostingClassifier): log-loss gradient boosting for binary and multiclass classification
- [XGBoostRegressor](xgboostRegressor): exact-greedy, regularized XGBoost regression
- [XGBoostClassifier](xgboostClassifier): exact-greedy, regularized XGBoost classification (binary and multiclass)

## Detailed module guide

### How to choose an algorithm

1. Use [Isolation Forest](iforest) when labels are unavailable and the main goal is outlier detection.
2. Use [RandomForestClassifier](randomForestClassifier) or [RandomForestRegressor](randomForestRegressor) when tree variance is high and nonlinear patterns matter.
3. Use [BaggingClassifier](baggingClassifier) when you want bootstrap aggregation around decision trees or a custom classifier.
4. Use [AdaBoost Classifier](adaboostClassifier) when a simple classifier misses hard examples.
5. Use [AdaBoost Regressor](adaboost) when a single regressor is too weak for the target pattern.
6. Use [GradientBoostingClassifier](gradientBoostingClassifier) or [GradientBoostingRegressor](gradientBoostingRegressor) when you want stage-wise boosting with fine control over shrinkage and depth.
7. Use [XGBoostClassifier](xgboostClassifier) or [XGBoostRegressor](xgboostRegressor) when you want regularized, exact-greedy boosting close to the `xgboost` library.

### JavaScript deployment notes

- Treat thresholds, estimator counts, and learning rates as product-level tuning decisions.
- Ensemble models often provide a good middle ground between simple baselines and more operationally complex systems.
- In JS applications, keep feature extraction and prediction close together so model output can flow directly into business logic.
