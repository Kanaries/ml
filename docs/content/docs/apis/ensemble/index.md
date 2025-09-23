---
title: Ensemble Algorithms in @kanaries/ml
description: Detailed API reference for Isolation Forest and AdaBoost ensemble learning algorithms in JavaScript and TypeScript.
---

# Ensemble algorithms

Ensemble methods in @kanaries/ml combine multiple weak learners to produce stronger, more resilient models for both regression and classification tasks. Use the links below to explore parameter definitions, method signatures, and implementation details for each estimator.

## Isolation Forest for anomaly detection

`Ensemble.IsolationForest` isolates anomalies by randomly partitioning the feature space. Shorter average path lengths indicate outliers, making this algorithm well-suited for fraud detection, observability metrics, and IoT monitoring in JavaScript environments.

**Highlights**

- Works with high-dimensional tabular data without heavy preprocessing.
- Offers configurable contamination rates to tune sensitivity.
- Runs in browsers or Node.js with minimal dependencies.

[Read the full Isolation Forest API reference](iforest.md) for constructor arguments, fit/predict usage, and scoring helpers.

## AdaBoostRegressor for gradient boosting

`Ensemble.AdaBoostRegressor` sequentially trains weak regressors and boosts their contributions to reduce error. It is ideal for modeling continuous targets where interpretability and responsiveness matter.

**Use cases**

- Forecasting metrics for product analytics dashboards.
- Enhancing baseline linear models with non-linear corrections.
- Building lightweight regressors that deploy quickly in serverless functions.

[Review the AdaBoost regressor documentation](adaboost.md) for hyperparameters and example workflows.

## AdaBoostClassifier for robust classification

`Ensemble.AdaBoostClassifier` focuses on misclassified samples in each iteration to improve predictive accuracy. Apply it to click-through rate prediction, churn modeling, or any binary/multi-class problem where you need fast inference in JavaScript.

**Key capabilities**

- Adjustable learning rates to balance speed and generalization.
- Support for sample weighting and class imbalance handling.
- Compatible with the preprocessing tools found in the [utils module](../utils/index.md).

[Explore the AdaBoost classifier guide](adaboostClassifier.md) for parameter tables, training instructions, and evaluation tips.

## Related resources

- Return to the [API overview](../index.md) to browse additional algorithm families.
- Follow the [Getting Started tutorial](../../index.mdx) to integrate ensembles into a new project.
- Experiment with datasets in the [`examples` folder](../../../../../examples) to see ensemble models in action.
