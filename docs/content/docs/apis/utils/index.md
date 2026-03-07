---
title: Utils
description: Utility functions for the ml library
---

- [asyncMode](asyncMode.md)

## How to use the Utils module in real projects

The Utils module includes workflow helpers that make JavaScript ML pipelines easier to train, evaluate, and ship.

### Selection checklist
1. Use asyncMode for non-blocking training or inference in UI-heavy browser applications.
2. Combine utilities with model modules to build consistent preprocessing and evaluation flows.
3. Standardize split logic and metric checks to keep experiments reproducible.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `ml utility functions javascript`
- `async machine learning browser`
- `typescript model selection helpers`

### Explore algorithms in this module
- [asyncMode](asyncMode)

