---
title: Missing-Value Imputation in JavaScript
description: Handle NaN missing values in tabular JavaScript and TypeScript ML pipelines with IterativeImputer (MICE) from @kanaries/ml in browser and Node.js.
---

## Module overview

Real-world CSV uploads, form telemetry, and sensor tables almost always arrive with missing values (`NaN`). Dropping every incomplete row wastes valuable training data and introduces selection bias, while naive global column-mean replacement flattens feature correlations and distorts linear or tree boundaries.

The **`Impute`** module provides scikit-learn-style transformer imputation that learns missing-value models strictly on training splits—preventing test-set leakage when placed inside a [`Pipeline`](/docs/apis/pipeline) or cross-validation loop.

## JavaScript implementation

`@kanaries/ml` provides [`Impute.IterativeImputer`](/docs/apis/impute/iterativeImputer) (multivariate imputation by chained equations / MICE) in pure TypeScript for browser and Node.js applications. Missing numeric entries are represented as standard JavaScript `NaN` values, and fitted imputers serialize cleanly alongside downstream estimators via `toJSON()` and [`loadModel()`](/docs/apis/serialization).

## Quick start example

```ts
import { Impute, Linear, Pipeline } from '@kanaries/ml';

const pipe = new Pipeline({
  steps: [
    ['imputer', new Impute.IterativeImputer({ maxIter: 10, tol: 1e-3 })],
    ['reg', new Linear.RidgeRegression({ alpha: 0.5 })],
  ],
});

pipe.fit([[0, 1], [1, 3], [2, NaN], [3, 7]], [2, 5, 8, 11]);
console.log(pipe.predict([[4, NaN]]));
```

## Choosing between `SimpleImputer` and `IterativeImputer`

| Strategy | Module | Best when |
| --- | --- | --- |
| Univariate fill (`mean`, `median`, `mostFrequent`, `constant`) | [`utils.Preprocessing.SimpleImputer`](/docs/apis/utils/preprocessing) | Missingness is sparse, columns are mostly independent, or latency budgets are tight |
| Multivariate chained regression (MICE) | [`Impute.IterativeImputer`](/docs/apis/impute/iterativeImputer) | Missing columns correlate with other features and preserving cross-column structure matters |

- Clamp imputed outputs to valid physical domains using `minValue` and `maxValue` (for example `minValue: 0` for strictly non-negative counts or prices).
- Always place imputation inside a [`Pipeline`](/docs/apis/pipeline) before cross-validation so training statistics never leak into held-out folds.
- **Related pages**: [IterativeImputer API Reference](/docs/apis/impute/iterativeImputer), [ML Pipelines](/docs/apis/pipeline), [Preprocessing Utilities](/docs/apis/utils/preprocessing).
