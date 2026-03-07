---
title: SVM
description: Learn how to use SVM algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [SVC](SVC)
- [NuSVC](NuSVC)
- [LinearSVC](LinearSVC)
- [LinearSVR](LinearSVR)

## How to use the SVM module in real projects

The SVM module offers margin-based models for classification and regression with strong performance on medium-sized structured datasets.

### Selection checklist
1. Use SVC or NuSVC for non-linear or margin-sensitive classification settings.
2. Use LinearSVC for large sparse feature spaces where linear boundaries are sufficient.
3. Use LinearSVR for regression with margin-based robustness to moderate noise.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `svm javascript`
- `linearsvc typescript`
- `support vector machine nodejs`

### Explore algorithms in this module
- [LinearSVC](LinearSVC)
- [LinearSVR](LinearSVR)
- [NuSVC](NuSVC)
- [SVC](SVC)

