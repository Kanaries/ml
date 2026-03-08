---
title: Lightweight Neural Models in JavaScript with @kanaries/ml
description: Explore Bernoulli RBM in JavaScript and TypeScript with @kanaries/ml for feature learning and latent representation workflows.
---

# Lightweight Neural Models in JavaScript

## Module overview

The Neural Network module currently focuses on lightweight representation learning rather than full deep learning stacks. It is useful when you want to learn latent features inside a JavaScript pipeline and then feed those features into downstream models or visual workflows.

This module is a strong fit when:

- your data can benefit from compact learned representations
- you want a JS-native feature-learning step before a classifier or regressor
- you are building experimentation or educational workflows around latent features

## JavaScript implementation

`@kanaries/ml` provides lightweight neural feature-learning utilities in JavaScript and TypeScript so teams can keep representation learning close to the rest of the application stack. This is especially useful when transformed features need to flow directly into additional JS models, UI-driven exploration, or Node.js preprocessing stages.

If someone searches for "Bernoulli RBM in JavaScript" or "feature learning in TypeScript", this module is the right entry point.

## Quick navigation

- [Bernoulli RBM](bernoulliRBM): learn latent binary features from binary or binarized inputs

## Detailed module guide

### How to choose an algorithm

1. Start with [Bernoulli RBM](bernoulliRBM) when your inputs are binary or can be meaningfully binarized.
2. Use the learned features as an experimental preprocessing step before downstream classifiers or regressors.
3. Compare downstream lift against raw features before adopting the representation in production.

### JavaScript deployment notes

- Keep datasets moderate in size when running iterative representation learning in the browser.
- Use this module mainly for preprocessing, feature learning, and experimentation rather than as a complete neural stack.
- Pair learned features with simpler downstream estimators when you want the rest of the pipeline to remain easy to operate.
