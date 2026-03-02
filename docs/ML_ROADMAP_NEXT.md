# ML Package Next-Step Roadmap

## Current State (March 2, 2026)
- Test health: `73/73` suites passing (`136` tests).
- Broad algorithm coverage already exists across:
  - Linear, tree, SVM, Bayes, clustering, manifold, decomposition, semi-supervised, ensemble.
- Recent progress completed:
  - `PolynomialRegression`
  - `StandardScaler` / `MinMaxScaler`
  - `KFold` / `crossValScore`
  - Expanded metrics (`precision`, `recall`, `f1`, `MSE`, `R2`)
- Remaining gaps are now mostly around:
  - Additional classical estimators.
  - API consistency and documentation depth.

## Priority Plan

### 1. Core Utilities (completed)
1. Add `StandardScaler` and `MinMaxScaler` to a preprocessing module.
2. Add reusable evaluation metrics (`accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `r2Score`, `meanSquaredError`).
3. Add `KFold` and `crossValScore` helpers to mirror sklearn workflows.

Why first:
- These are used across almost all algorithms.
- They improve user experience more than adding one-off estimators.

### 2. High-Impact Missing Estimators (next)
1. `RandomForestClassifier`
2. `RandomForestRegressor`
3. `RidgeRegression` and `LassoRegression`

Why second:
- These cover common production baselines and align with sklearn expectations.

### 3. API and Compatibility Improvements (next)
1. Normalize constructor signatures (`props` objects, defaults, validation).
2. Add optional `predictProba` where classification models support it.
3. Add `getParams` and `setParams` on estimators for pipeline-style usage.

### 4. Documentation and Examples (in progress)
1. Add comparison examples: browser + Node usage for each major family.
2. Add performance notes for dataset size ranges.
3. Add migration-style docs from sklearn to `@kanaries/ml`.

## Suggested Work Sequence (short horizon)
1. `RandomForestClassifier` based on current tree code reuse.
2. `RandomForestRegressor`.
3. `RidgeRegression` and `LassoRegression`.
4. Add docs pages for preprocessing, model selection, and expanded metrics.

## Success Criteria
- All new utilities have:
  - deterministic unit tests.
  - sklearn-compare tests where feasible.
  - top-level exports and README/docs coverage.
- CI remains green with no regressions in existing suites.
