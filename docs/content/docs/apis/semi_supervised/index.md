---
title: SemiSupervised
description: Learn how to use SemiSupervised algorithms in @kanaries/ml for JavaScript and TypeScript machine learning projects.
---

- [LabelPropagation](labelPropagation)
- [LabelSpreading](labelSpreading)

## How to use the SemiSupervised module in real projects

The SemiSupervised module helps when labeled data is limited but unlabeled samples are abundant.

### Selection checklist
1. Use LabelPropagation for graph-style transductive learning when neighborhood structure is reliable.
2. Use LabelSpreading for smoother label diffusion with stronger regularization.
3. Benchmark against a supervised-only baseline to confirm unlabeled data is adding value.

### Common implementation workflow
1. Start from a simple baseline in this module and evaluate on a holdout split.
2. Compare at least one alternative algorithm from this module before locking production defaults.
3. Pair model quality metrics with runtime constraints (latency, memory, bundle size).

### Common search intents
- `semi supervised learning javascript`
- `label propagation typescript`
- `low label ml nodejs`

### Explore algorithms in this module
- [labelPropagation](labelPropagation)
- [labelSpreading](labelSpreading)

