# Phase 1 CodeX Review — Findings & Resolutions

Reviewer: codex-cli 0.144.1 (independent), scope: commits `ff6fcc2` + `8e9b8c1`
(Pipeline/ColumnTransformer/FeatureUnion, datasets, metrics fill-out,
preprocessingExtra, modelSelectionExtra, e2e tutorial test). Date: 2026-07-11.

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| 1 | P1 | `crossValidate`/`learningCurve`/`validationCurve` defaulted to plain `KFold` even for classifiers — label-ordered data yields single-class training folds (degenerate or crashing fits). sklearn stratifies for classifiers. | **Fixed** — integer/undefined `cv` resolves to `StratifiedKFold` when the estimator (or a pipeline's final step) is a `ClassifierBase`. |
| 2 | P1 | `StratifiedShuffleSplit` computed class allocations once with deterministic tie-breaking and reused them for every split — the same class systematically got the remainder slot, biasing repeated-CV scores. | **Fixed** — allocation happens per split with RNG-jittered tie-breaking (`allocateProportionally` gained an optional RNG). |
| 3 | P1 | `ColumnTransformer` never recorded the fitted feature count; transform silently accepted inference data with a different column count (schema drift → plausible wrong outputs). | **Fixed** — `nFeaturesIn` recorded at fit, transform throws on mismatch. |
| 4 | P1 | `PowerTransformer` hard-clamped the lambda MLE search to `[-5, 5]`; near-constant positive data (true λ ≈ −38) got boundary values and materially wrong transforms. sklearn's Brent search is unbounded. | **Fixed** — bracket expands (×4, up to 6 rounds) while the optimum lands on a boundary. |
| 5 | P1 | `KNNImputer.fit` threw when a training feature was entirely missing — breaking ordinary fold-local imputation on sparse features. sklearn drops such features. | **Fixed** — all-missing features are dropped at fit; transform outputs only the kept columns (sklearn behavior). |
| 6 | P2 | `ColumnTransformer` accepted transformer names containing `__`, making those entries unaddressable via nested `setParams`. | **Fixed** — names with `__` rejected (parity with Pipeline/FeatureUnion). |
| 7 | P2 | Binary ROC-AUC with single-class labels silently returned 0 instead of failing (sklearn treats it as undefined). | **Fixed** — `rocCurve`/`rocAucScore` throw when only one class is present. |
| 8 | P2 | `QuantileTransformer` allowed `nQuantiles > subsample`, fabricating quantiles from fewer observations than requested. | **Fixed** — constructor rejects the combination. |
| 9 | P2 | Absolute (≥1) `testSize`/`trainSize` values were silently floored (`1.9` → 1 sample). sklearn requires integers. | **Fixed** — non-integer absolute sizes throw, in both `trainTestSplit` and the shuffle splitters. |

Regression coverage: `src/__test__/phase1.review-fixes.test.ts` (one block per finding).
All fixes verified by the full suite (`yarn jest`) and typecheck.
