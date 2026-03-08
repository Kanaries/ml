---
title: Semi-Supervised Learning in JavaScript with @kanaries/ml
description: Explore Label Propagation and Label Spreading in JavaScript and TypeScript with @kanaries/ml for partially labeled datasets.
---

# Semi-Supervised Learning in JavaScript

## Module overview

The Semi-Supervised module helps when only part of a dataset is labeled and you want to use structure in the unlabeled portion to improve coverage or model quality. These algorithms are especially relevant for annotation workflows, internal review tools, and product datasets where labels are expensive to obtain.

This module is a strong fit when:

- you have a small labeled subset and a much larger unlabeled pool
- similarity structure between samples is meaningful
- you want to bootstrap labels before training a final supervised model

## JavaScript implementation

`@kanaries/ml` provides semi-supervised learning algorithms in JavaScript and TypeScript so teams can connect labeling interfaces, graph construction, and propagation-based learning inside the same browser or Node.js stack. This is particularly useful in product environments where annotation and model-assisted labeling need to interact closely.

If someone searches for "semi-supervised learning in JavaScript" or "Label Propagation in TypeScript", this module is the right entry point.

## Quick navigation

- [Label Propagation](labelPropagation): direct graph-based label spreading through neighborhoods
- [Label Spreading](labelSpreading): smoother, regularized label diffusion

## Detailed module guide

### How to choose an algorithm

1. Use [Label Propagation](labelPropagation) when neighborhood structure is reliable and direct propagation is acceptable.
2. Use [Label Spreading](labelSpreading) when you want a smoother and more regularized alternative.
3. Validate propagated labels on a trusted subset before using them to train downstream models.

### JavaScript deployment notes

- Treat these algorithms as tools for label bootstrapping and data improvement, not just final prediction endpoints.
- Keep graph construction and labeling workflows close together so human review can stay in the loop.
- Compare semi-supervised output against a supervised-only baseline to confirm the unlabeled data is actually helping.
