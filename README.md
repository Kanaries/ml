# @kanaries/ml

![License](https://img.shields.io/github/license/kanaries/ml?color=%23FF7575)
![CI](https://github.com/kanaries/ml/actions/workflows/ci.yml/badge.svg)

**@kanaries/ml** is a JavaScript/TypeScript library that provides a set of machine learning algorithms with an API similar to scikit-learn. It works in both browsers and Node.js environments.

## Features

- Classification and regression trees
- k-nearest neighbors
- Support vector machines
- Clustering algorithms (KMeans, DBSCAN, OPTICS, Mean Shift, HDBSCAN)
- Dimensionality reduction (PCA)
- Manifold learning (t-SNE)
- Basic linear algebra utilities

## Installation

```sh
# using yarn
yarn add @kanaries/ml

# or npm
npm install @kanaries/ml
```

## Quick Start

```js
import { Neighbors } from '@kanaries/ml';

const trainX = [
    [0.12, 0.2, /* ... */ 0.2],
    [0.21, 0.3, /* ... */ 0.2],
];
const trainY = [0, 1];

const knn = new Neighbors.KNearstNeighbors(3, 'distance', '2-norm');
knn.fit(trainX, trainY);

const testX = [
    [0.52, 0.72, /* ... */ 0.24],
    [0.11, 0.98, /* ... */ 0.32],
];
const result = knn.predict(testX);
console.log(result);
```

## Development

```sh
# Install dependencies
yarn

# Run tests
npm run test

# Build the library
yarn build

# Start the example development server
yarn dev
```

