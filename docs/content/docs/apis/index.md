---
title: JavaScript Machine Learning API Reference
description: Explore the @kanaries/ml API for clustering, classification, anomaly detection, and more in JavaScript and TypeScript.
---

# @kanaries/ml API Reference

This catalog describes every module available in @kanaries/ml so you can quickly locate the algorithm or utility that fits your JavaScript or TypeScript project. Each section below links to a detailed page covering parameters, usage patterns, and practical tips.

- **[Clusters](clusters/index.md)** – Implement k-means, DBSCAN, spectral clustering, and other unsupervised techniques for segmentation and feature discovery.
- **[Decomposition](decomposition/index.md)** – Run dimensionality reduction methods such as PCA and SVD directly in the browser or Node.js.
- **[Ensemble](ensemble/index.md)** – Combine multiple estimators including Isolation Forest and AdaBoost for robust anomaly detection and boosting workflows.
- **[Linear](linear/index.md)** – Access regression and classification algorithms like Linear Regression, Lasso, and Logistic Regression with sklearn-like APIs.
- **[Manifold](manifold/index.md)** – Visualize high-dimensional data using t-SNE, Isomap, and related manifold learning approaches.
- **[Neighbors](neighbors/index.md)** – Use k-nearest neighbors for search, recommendation, and classification tasks.
- **[SVM](svm/index.md)** – Train support vector machines for high-margin classification and regression scenarios.
- **[Tree](tree/index.md)** – Build decision trees and random forests for interpretable models and tabular data analysis.
- **[Bayes](bayes/index.md)** – Apply naive Bayes and probabilistic models to text classification, spam detection, and more.
- **[NeuralNetwork](neural_network/index.md)** – Prototype lightweight neural networks tailored for in-browser inference.
- **[Utils](utils/index.md)** – Leverage preprocessing helpers, metrics, and shared utilities to streamline your ML pipelines.
- **[SemiSupervised](semi_supervised/index.md)** – Combine labeled and unlabeled data to boost performance in low-label environments.

Looking for inspiration? The [Getting Started guide](../index.mdx) and [examples directory](../../../../examples) provide end-to-end workflows you can adapt for your application.

## How to work with this API catalog

Use this page to map business problems to the right @kanaries/ml algorithm family before diving into constructor parameters.

### Selection checklist
1. Identify your task type first: classification, regression, clustering, dimensionality reduction, anomaly detection, or utility workflow support.
2. Start with one simple baseline model and one stronger alternative from the same module.
3. Evaluate model quality together with runtime constraints such as browser latency, Node.js throughput, and bundle size.

### Practical implementation flow
1. Open the module page that matches your task and read the algorithm-specific "when to use" guidance.
2. Build a minimal `fit`/`predict` pipeline and validate it on holdout data.
3. Tune model settings and deployment strategy (browser worker vs Node.js service) before production rollout.

### Search intents this page targets
- `javascript machine learning api reference`
- `typescript machine learning algorithms`
- `scikit-learn style api for javascript`
- `@kanaries/ml algorithm documentation`
