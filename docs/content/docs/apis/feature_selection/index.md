---
title: Feature Selection in JavaScript with @kanaries/ml
description: Prune noisy and redundant tabular columns in JavaScript and TypeScript using SelectFromModel, RFE, RFECV, chi2, ANOVA F, and mutual information in @kanaries/ml.
---

## Module overview

High-dimensional tabular datasets frequently contain noisy, constant, or redundant columns that inflate training latency, enlarge serialized browser model payloads, and cause overfitting. **Feature selection** identifies and keeps only the most informative columns before or during model training—yielding smaller, faster, and easier-to-interpret models.

`@kanaries/ml` organizes feature selection into two complementary approaches:

1. **Filter methods ([Univariate Scores](/docs/apis/feature_selection/univariate))**: score every feature independently against target `y` in a single fast pass—ideal for rapid exploratory screening or wide feature tables.
   - `chi2(X, y)` — chi-square dependence for non-negative count/frequency features.
   - `fClassif(X, y)` — one-way ANOVA F-statistic for continuous numeric features in classification.
   - `mutualInfoClassif(X, y)` / `mutualInfoRegression(X, y)` — detects arbitrary non-linear dependence via k-NN entropy estimation.
2. **Wrapper & embedded selectors ([Model-Based & Recursive Selectors](/docs/apis/feature_selection/selectors))**: leverage a supervised estimator's learned `coef` or `featureImportances` inside a [`Pipeline`](/docs/apis/pipeline):
   - `SelectFromModel` — keeps features exceeding an importance threshold in one fit.
   - `RFE` — iteratively removes the weakest features until `nFeaturesToSelect` remain.
   - `RFECV` — runs cross-validated RFE to pick the optimal feature count automatically.

## JavaScript implementation

Import every selector and scoring function from `import { FeatureSelection } from '@kanaries/ml'`. Selectors extend `BaseEstimator`, clone the supplied prototype estimator safely without mutating it, compose cleanly inside [`Pipeline`](/docs/apis/pipeline), and serialize via `toJSON()` / [`loadModel()`](/docs/apis/serialization).

## Quick start example

```ts
import { FeatureSelection, Tree } from '@kanaries/ml';

const X = [[0, 1, 9], [1, 0, 9], [8, 1, 9], [9, 0, 9]];
const y = [0, 0, 1, 1];

const [fScores] = FeatureSelection.fClassif(X, y);
const rfe = new FeatureSelection.RFE({
  estimator: new Tree.DecisionTreeClassifier({ randomState: 42 }),
  nFeaturesToSelect: 1,
});
rfe.fit(X, y);
console.log({ fScores, selectedIndices: rfe.getSupport(true) });
```

## Choosing a feature selection strategy

- **Need speed on wide matrices?** Score columns first with [`fClassif`, `chi2`, or `mutualInfoClassif`](/docs/apis/feature_selection/univariate).
- **Want single-fit embedded selection inside a Pipeline?** Use [`SelectFromModel`](/docs/apis/feature_selection/selectors) paired with [`Linear.LassoRegression`](/docs/apis/linear/lassoRegression) or [`Ensemble.ExtraTreesClassifier`](/docs/apis/ensemble/extraTreesClassifier).
- **Want optimal subset size validated by CV?** Use [`RFECV`](/docs/apis/feature_selection/selectors).
