---
title: Naive Bayes in JavaScript with @kanaries/ml
description: Explore Bernoulli Naive Bayes and Categorical Naive Bayes in JavaScript and TypeScript with @kanaries/ml for fast probabilistic classification.
---

# Naive Bayes in JavaScript

## Module overview

The Bayes module in `@kanaries/ml` focuses on lightweight probabilistic classifiers for binary and categorical inputs. These models are often useful as fast baselines because they train quickly, infer quickly, and are easy to reason about when you want probability-oriented outputs.

This module is a strong fit when:

- features are binary indicators or categorical IDs
- you want a simple model before moving to linear, tree, or ensemble methods
- low-latency inference matters in browser or Node.js applications

## JavaScript implementation

`@kanaries/ml` brings naive Bayes estimators into JavaScript and TypeScript so teams can keep feature encoding, prediction logic, and application code in the same runtime. This is especially useful for frontend-heavy workflows, text or flag-based classification, and product logic that benefits from lightweight probability estimates.

If someone searches for "naive Bayes in JavaScript" or "CategoricalNB in TypeScript", this module page should make it clear that `@kanaries/ml` provides JS-native implementations with a familiar estimator workflow.

## Quick navigation

- [BernoulliNB](bernoulliNB): best for boolean or presence/absence features
- [CategoricalNB](categoricalNB): best for integer-encoded categorical inputs

## Detailed module guide

### How to choose an algorithm

1. Use [BernoulliNB](bernoulliNB) when features represent yes/no states, token presence, clicks, or flags.
2. Use [CategoricalNB](categoricalNB) when each feature is a discrete category rather than a continuous value.
3. Compare calibration and class-specific metrics before deciding whether to stay with Bayes models or move to a more expressive classifier.

### JavaScript deployment notes

- Keep feature encoding rules stable between training and inference.
- These models are especially attractive in JS applications because they are easy to run close to product logic.
- Start here when you need a fast probabilistic baseline and low operational complexity.
