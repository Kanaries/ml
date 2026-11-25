---
title: Text Feature Extraction in JavaScript with @kanaries/ml
description: Convert raw text and dictionaries into sparse CSR count and TF-IDF matrices in JavaScript and TypeScript with @kanaries/ml for browser and Node.js NLP.
---

## Module overview

Machine learning models operate on numeric feature matrices, yet application data frequently arrives as raw strings (support tickets, search queries, product descriptions, user reviews) or semi-structured JSON objects. Materializing a dense `nDocuments × nVocabulary` array wastes memory rapidly once vocabularies reach thousands of terms.

The **`FeatureExtraction`** module tokenizes text, weights informative terms, hashes high-cardinality categories, and outputs memory-efficient [`CSRMatrix`](/docs/apis/data) sparse matrices that flow directly through [`Pipeline`](/docs/apis/pipeline) into sparse-aware estimators like [`Bayes.MultinomialNB`](/docs/apis/bayes/multinomialNB), [`Bayes.ComplementNB`](/docs/apis/bayes/complementNB), and [`Decomposition.LatentDirichletAllocation`](/docs/apis/decomposition/latentDirichletAllocation).

## JavaScript implementation

`@kanaries/ml` provides the scikit-learn `feature_extraction` family in pure TypeScript for browser and Node.js runtimes—with zero Python backend or native tokenizer binaries:

| Transformer | Input type | Stores vocabulary? | Best for |
| --- | --- | --- | --- |
| [`CountVectorizer`](/docs/apis/feature_extraction/countVectorizer) | `string[]` | Yes (`Map`) | Bag-of-words / n-gram counts, LDA topic modeling |
| [`TfidfTransformer`](/docs/apis/feature_extraction/tfidfTransformer) | Dense or `CSRMatrix` counts | No (stores `idf`) | Re-weighting precomputed count matrices |
| [`TfidfVectorizer`](/docs/apis/feature_extraction/tfidfVectorizer) | `string[]` | Yes (`Map` + `idf`) | End-to-end text classification & search baselines |
| [`HashingVectorizer`](/docs/apis/feature_extraction/hashingAndDictionaries) | `string[]` | No (stateless MurmurHash3) | Streaming or bounded-memory wide text workloads |
| [`DictVectorizer` / `FeatureHasher`](/docs/apis/feature_extraction/hashingAndDictionaries) | `Record<string, number \| string>[]` | Yes / No | Converting heterogeneous JS objects into matrices |

## Quick start

```ts
import { Bayes, FeatureExtraction, Pipeline } from '@kanaries/ml';

const model = new Pipeline({
  steps: [
    ['tfidf', new FeatureExtraction.TfidfVectorizer({ ngramRange: [1, 2] })],
    ['classifier', new Bayes.MultinomialNB()],
  ],
});

model.fit(['red apple sweet', 'blue ocean deep'], [0, 1]);
console.log(model.predict(['sweet apple']));
```

## Choosing a feature extractor

- Start with **[`TfidfVectorizer`](/docs/apis/feature_extraction/tfidfVectorizer)** for standard text classification or document similarity in browser and Node.js pipelines.
- Use **[`CountVectorizer`](/docs/apis/feature_extraction/countVectorizer)** when feeding count-based generative models like [`LatentDirichletAllocation`](/docs/apis/decomposition/latentDirichletAllocation).
- Switch to **[`HashingVectorizer` or `FeatureHasher`](/docs/apis/feature_extraction/hashingAndDictionaries)** when vocabulary size is unbounded or serialized JSON model size must stay fixed.
- Use **[`DictVectorizer`](/docs/apis/feature_extraction/hashingAndDictionaries)** to one-hot encode mixed numeric and categorical JavaScript objects with invertible feature names.
