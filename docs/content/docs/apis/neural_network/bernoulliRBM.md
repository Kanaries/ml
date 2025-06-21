---
title: BernoulliRBM
description: API reference for BernoulliRBM
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
