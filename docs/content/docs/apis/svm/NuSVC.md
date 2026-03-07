---
title: NuSVC
description: API and practical guide for NuSVC in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SVM.NuSVC

Variant of `SVC` that uses the `nu` parameter to control the fraction of support vectors and the margin errors.

```ts
interface NuSVCProps {
    nu?: number;
}
constructor(props: NuSVCProps = {})
```

### Parameters
- `nu` (number, default `0.5`): trade-off between the number of support vectors and training errors
- `C` (number, default `1/nu`): regularization strength
- `maxIter` (number, default `100`): maximum iterations for training
- `learningRate` (number, default `0.01`): step size for gradient descent
- `kernel` ('linear' | 'rbf', default `'rbf'`): kernel type
- `gamma` (number, default `1`): RBF kernel coefficient

NuSVC shares the same usage as `SVC` but uses the `nu` parameter instead of `C`.

## Practical guide: NuSVC in JavaScript and TypeScript

NuSVC provides SVM classification with nu-based control over support vector fraction and margin errors.

### When to use NuSVC
- You prefer `nu` constraints over classic `C` tuning behavior.
- Dataset boundaries require flexible kernel-based classification.
- You need explicit control over support-vector complexity.

### Implementation workflow
1. Choose kernel and initialize a conservative `nu` value.
2. Fit NuSVC and inspect validation accuracy and support vector count.
3. Tune kernel and nu jointly for generalization and runtime limits.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `NuSVC JavaScript`
- `NuSVC TypeScript`
- `NuSVC browser machine learning`
- `@kanaries/ml NuSVC`

