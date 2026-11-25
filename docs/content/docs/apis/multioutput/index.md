---
title: Multi-Output Machine Learning in JavaScript
description: Predict multiple correlated binary labels or numeric targets in JavaScript and TypeScript using ClassifierChain and RegressorChain from @kanaries/ml.
---

## Module overview

Standard classifiers and regressors predict a single scalar target `y[i]` per observation. Many practical problems instead require predicting **multiple correlated outputs simultaneously**—such as multi-label tagging (assigning several topic tags to one article) or joint forecasting (predicting both `revenue` and `order_count` from the same customer features).

Fitting completely independent models per target ignores relationships *between* the targets themselves. **Chain meta-estimators** preserve those correlations by ordering targets into a sequence and feeding earlier target predictions forward as extra input features for subsequent models in the chain.

## JavaScript implementation

`@kanaries/ml` provides two scikit-learn-compatible chain wrappers under `import { MultiOutput } from '@kanaries/ml'` in pure TypeScript for browser and Node.js runtimes:

| Estimator | Target matrix `Y` | Base estimator | Use case |
| --- | --- | --- | --- |
| [`MultiOutput.ClassifierChain`](/docs/apis/multioutput/classifierChain) | Binary `0/1` matrix `[nSamples][nLabels]` | Any registered classifier | Multi-label classification where labels co-occur or exclude one another |
| [`MultiOutput.RegressorChain`](/docs/apis/multioutput/regressorChain) | Continuous matrix `[nSamples][nTargets]` | Any registered regressor | Multi-target regression where outputs share strong correlations |

Both wrappers clone the prototype estimator safely, support explicit or seeded random target ordering (`order`), optionally use out-of-fold cross-validation (`cv`) to prevent target leakage while building augmented training columns, and serialize cleanly via `toJSON()` / [`loadModel()`](/docs/apis/serialization).

## Quick start example

```ts
import { Bayes, MultiOutput } from '@kanaries/ml';

const chain = new MultiOutput.ClassifierChain({
  estimator: new Bayes.GaussianNB(),
  order: [0, 1],
});
chain.fit([[0], [1], [2]], [[0, 0], [0, 1], [1, 1]]);
console.log(chain.predict([[1.5]]));
```

## Ordering and cross-validation guidance

- **Choosing `order`**: Place easier-to-predict or causally upstream targets first in the chain so downstream links receive cleaner features. When no natural causal order exists, set `order: 'random'` with a fixed `randomState` or compare multiple orders via cross-validation.
- **Enabling `cv`**: Setting `cv: 3` or `cv: 5` trains downstream links on out-of-fold upstream predictions instead of ground-truth targets, reducing train/inference mismatch.

## Related pages

- [ClassifierChain API Reference](/docs/apis/multioutput/classifierChain)
- [RegressorChain API Reference](/docs/apis/multioutput/regressorChain)
- [Multiclass Wrappers (`OneVsRest` / `OneVsOne`)](/docs/apis/multiclass)
