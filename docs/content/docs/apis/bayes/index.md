---
title: Bayes
description: Learn how to use Bayes algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [BernoulliNB](bernoulliNB)
- [CategoricalNB](categoricalNB)

## How to use the Bayes module in real projects

The Bayes module is ideal for fast probabilistic baselines in JavaScript, especially when you need interpretable class probabilities and low-latency inference.

### Selection checklist
1. Start with BernoulliNB for binary/count-like feature presence signals (for example, flags or bag-of-words indicators).
2. Use CategoricalNB when each feature is a discrete category index instead of a continuous value.
3. Evaluate probability calibration and confusion matrix metrics before moving to more complex models.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `naive bayes javascript`
- `categorical naive bayes typescript`
- `bernoulli nb browser ml`

### Explore algorithms in this module
- [bernoulliNB](bernoulliNB)
- [categoricalNB](categoricalNB)

