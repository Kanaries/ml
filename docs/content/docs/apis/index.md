---
title: JavaScript Machine Learning APIs with @kanaries/ml
description: Explore machine learning APIs in JavaScript and TypeScript with @kanaries/ml, including clustering, classification, anomaly detection, dimensionality reduction, and workflow utilities.
---

# JavaScript Machine Learning APIs

## Module overview

This page is the entry point to the full `@kanaries/ml` API catalog. It helps JavaScript and TypeScript teams choose the right algorithm family before diving into individual estimators such as K-Means, Logistic Regression, Isolation Forest, PCA, or k-Nearest Neighbors.

Use this catalog when you already know the kind of problem you need to solve, such as:

- classification or regression on tabular data
- clustering or segmentation without labels
- anomaly detection on product, telemetry, or transaction data
- dimensionality reduction for embeddings, charts, or preprocessing
- utility workflows such as non-blocking execution in browser or Node.js apps

## JavaScript implementation

`@kanaries/ml` provides a scikit-learn-like machine learning API for JavaScript and TypeScript so teams can build ML workflows directly in browser applications and Node.js services. Instead of treating ML as a separate Python-only layer, you can keep feature engineering, inference logic, interactive visualizations, and product code inside the same JS stack.

This is especially useful when someone searches for "machine learning in JavaScript", "TypeScript machine learning APIs", or "scikit-learn for JavaScript" and needs a practical module map rather than a single algorithm page.

## Quick navigation

- **[Clusters](/docs/apis/clusters)**: segment unlabeled data with K-Means, HDBSCAN, Mean Shift, OPTICS, and initialization helpers.
- **[Decomposition](/docs/apis/decomposition)**: reduce dimensions with PCA, Sparse PCA, and Truncated SVD.
- **[Ensemble](/docs/apis/ensemble)**: use Isolation Forest, AdaBoost, random forest, and bagging models for anomaly detection, classification, and regression.
- **[Linear](/docs/apis/linear)**: start with linear regression, logistic regression, regularized regression, and linear classification baselines.
- **[Metrics](/docs/apis/metrics)**: evaluate classification, regression, clustering, curves, and distance functions.
- **[Manifold](/docs/apis/manifold)**: build lower-dimensional embeddings for visualization and neighborhood analysis.
- **[Neighbors](/docs/apis/neighbors)**: run k-nearest neighbors and fast nearest-neighbor search structures.
- **[SVM](/docs/apis/svm)**: train support vector models for classification and regression.
- **[Tree](/docs/apis/tree)**: use interpretable decision tree and extra tree models.
- **[Bayes](/docs/apis/bayes)**: apply naive Bayes models to binary or categorical features.
- **[Neural Network](/docs/apis/neural_network)**: learn compact representations with Bernoulli RBM.
- **[Semi-Supervised](/docs/apis/semi_supervised)**: spread labels through partially labeled datasets.
- **[Utils](/docs/apis/utils)**: use preprocessing, sampling, model selection, statistics, and async workflow helpers.
- **[Algebra](/docs/apis/algebra)**: use lightweight matrix helpers such as transpose, determinants, and inverse.
- **[KMath](/docs/apis/KMath)**: compute lightweight descriptive statistics.

## Detailed module guide

### How to choose a module

1. Start from the task type: classification, regression, clustering, anomaly detection, embedding, or workflow support.
2. Open the matching module page and compare one simple baseline against one stronger alternative.
3. Read the algorithm-specific guidance to balance model quality with browser latency, Node.js throughput, and implementation complexity.

### Recommended learning path

1. If you are new to the library, start with [Linear](/docs/apis/linear), [Tree](/docs/apis/tree), or [Clusters](/docs/apis/clusters).
2. Move to [Ensemble](/docs/apis/ensemble) and [SVM](/docs/apis/svm) when simple baselines are not expressive enough.
3. Use [Metrics](/docs/apis/metrics), [Decomposition](/docs/apis/decomposition), [Manifold](/docs/apis/manifold), and [Utils](/docs/apis/utils) to improve evaluation, preprocessing, visualization, and application integration.
