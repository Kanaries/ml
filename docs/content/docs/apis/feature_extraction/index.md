---
title: Text Feature Extraction in JavaScript with @kanaries/ml
description: Convert raw text into sparse count and TF-IDF matrices in JavaScript or TypeScript for browser and Node.js machine-learning pipelines.
---

# Text Feature Extraction in JavaScript

## Module overview

Text estimators need numeric features, but materializing a dense document-by-vocabulary matrix wastes memory. The `FeatureExtraction` module tokenizes strings and returns `CSRMatrix` sparse matrices that can flow through `Pipeline` and sparse-aware Naive Bayes estimators.

## JavaScript implementation

`@kanaries/ml` brings the familiar scikit-learn text workflow to browser and Node.js code: use `CountVectorizer` for token counts, `TfidfTransformer` for weighting existing count matrices, or `TfidfVectorizer` for both steps together.

## Quick start

```ts
import { Bayes, FeatureExtraction, Pipeline } from '@kanaries/ml';

const model = new Pipeline({ steps: [
  ['tfidf', new FeatureExtraction.TfidfVectorizer()],
  ['classifier', new Bayes.MultinomialNB()],
] });

model.fit(['red apple sweet', 'blue ocean deep'], [0, 1]);
model.predict(['sweet apple']);
```

## Choose an API

- [CountVectorizer](countVectorizer): sparse token and n-gram counts.
- [TfidfTransformer](tfidfTransformer): TF-IDF weighting for an existing count matrix.
- [TfidfVectorizer](tfidfVectorizer): one-step text-to-TF-IDF conversion.
