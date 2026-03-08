---
title: Tree Models in JavaScript with @kanaries/ml
description: Explore decision trees and extra trees in JavaScript and TypeScript with @kanaries/ml for interpretable tabular classification and regression.
---

# Tree Models in JavaScript

## Module overview

The Tree module provides interpretable tree-based models for tabular classification and regression. These algorithms are useful when you want rule-like behavior, easy debugging, and models whose decisions can be traced through human-readable splits.

This module is a strong fit when:

- explainability matters alongside predictive quality
- tabular data contains non-linear feature interactions
- you want to compare standard tree behavior with more randomized extra tree variants

## JavaScript implementation

`@kanaries/ml` provides tree models in JavaScript and TypeScript so teams can keep interpretable prediction logic close to product code, browser demos, and Node.js APIs. This is especially useful when engineers or stakeholders need to inspect why a model made a decision, not just what it predicted.

If someone searches for "decision tree in JavaScript" or "extra tree classifier in TypeScript", this module is the right entry point.

## Quick navigation

- [Decision Tree Classifier](decisionTreeClassifier)
- [Decision Tree Regressor](decisionTreeRegressor)
- [Extra Tree Classifier](extraTreeClassifier)
- [Extra Tree Regressor](extraTreeRegressor)

## Detailed module guide

### How to choose an algorithm

1. Use decision tree variants when interpretability and clear decision paths matter most.
2. Use extra tree variants when you want a more randomized tree baseline.
3. Control depth and split settings carefully because tree models can overfit quickly on smaller datasets.

### JavaScript deployment notes

- Tree models are especially practical in product code because their behavior is straightforward to inspect.
- Use them as strong tabular baselines before moving to boosted ensembles or kernel methods.
- Keep model constraints simple and explicit so debugging remains easy.
