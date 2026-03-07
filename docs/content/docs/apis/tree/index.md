---
title: Tree
description: Learn how to use Tree algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [DecisionTreeClassifier](decisionTreeClassifier.md)

- [ExtraTreeClassifier](extraTreeClassifier.md)

- [DecisionTreeRegressor](decisionTreeRegressor.md)
- [ExtraTreeRegressor](extraTreeRegressor.md)

## How to use the Tree module in real projects

The Tree module provides interpretable decision rules for classification and regression on heterogeneous tabular features.

### Selection checklist
1. Use DecisionTree variants when model explainability and rule extraction matter.
2. Use ExtraTree variants when you want stronger randomization and variance reduction.
3. Control depth, split criteria, and validation performance to avoid overfitting.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `decision tree javascript`
- `extra tree classifier typescript`
- `interpretable ml nodejs`

### Explore algorithms in this module
- [decisionTreeClassifier](decisionTreeClassifier)
- [decisionTreeRegressor](decisionTreeRegressor)
- [extraTreeClassifier](extraTreeClassifier)
- [extraTreeRegressor](extraTreeRegressor)

