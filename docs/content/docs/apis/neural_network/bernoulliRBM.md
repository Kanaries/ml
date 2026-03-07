---
title: BernoulliRBM
description: API and practical guide for BernoulliRBM in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# NeuralNetwork.BernoulliRBM

```ts
constructor(
    nComponents: number = 256,
    learningRate: number = 0.1,
    batchSize: number = 10,
    nIter: number = 10
)
```

A Restricted Boltzmann Machine with binary visible units and hidden units. The model is trained with contrastive divergence.

### Methods
- `fit(X: number[][]): void`
- `partialFit(X: number[][]): void`
- `transform(X: number[][]): number[][]`
- `fitTransform(X: number[][]): number[][]`
- `gibbs(V: number[][]): number[][]`

### Example
```ts
const rbm = new BernoulliRBM({ nComponents: 2, nIter: 20 });
rbm.fit(data);
const h = rbm.transform(data);
```

## Practical guide: BernoulliRBM in JavaScript and TypeScript

BernoulliRBM learns latent binary representations that can improve downstream supervised model quality.

### When to use BernoulliRBM
- You work with binary-valued or binarized feature inputs.
- Feature learning can improve separability for a later classifier.
- You need compact latent features in a JavaScript pipeline.

### Implementation workflow
1. Prepare binary feature matrix and configure hidden-unit size.
2. Fit RBM and generate transformed latent representations.
3. Train downstream models on latent features and compare lift.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `BernoulliRBM JavaScript`
- `BernoulliRBM TypeScript`
- `BernoulliRBM browser machine learning`
- `@kanaries/ml BernoulliRBM`

