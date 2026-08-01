---
title: Multi-Output Machine Learning in JavaScript
description: Predict multiple related labels or numeric targets with chain estimators in browser and Node.js using @kanaries/ml.
---

# Multi-Output Machine Learning in JavaScript

Multi-output models predict several targets for each input row. Independent wrappers are useful when targets do not interact; classifier and regressor chains additionally feed earlier predictions into later members to model target dependencies.

`@kanaries/ml` exposes these estimators under `MultiOutput` with serializable, scikit-learn-style APIs for browser and Node.js.

- [ClassifierChain](classifierChain) for related binary labels.
- [RegressorChain](regressorChain) for related numeric outputs.
