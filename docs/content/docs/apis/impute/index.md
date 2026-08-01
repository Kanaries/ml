---
title: Missing-Value Imputation in JavaScript
description: Impute correlated numeric features in browser and Node.js with the @kanaries/ml JavaScript IterativeImputer implementation.
---

# Missing-Value Imputation in JavaScript

Missing values can break estimators or bias a complete-case analysis. The `Impute` module supplies transformer-style preprocessing that fits only on training data and can be placed inside repeatable JS pipelines.

Use [IterativeImputer](iterativeImputer) when features predict one another and simple column means discard too much structure.
