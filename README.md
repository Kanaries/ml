# @kanaries/ml

![](https://img.shields.io/github/license/kanaries/ml?color=%23FF7575)
![CI](https://github.com/kanaries/ml/actions/workflows/ci.yml/badge.svg)

machine learning lib in javascript

## Usage

```sh
npm i --save @kanaries/ml
```

examples:
```js
import { Neighbors } from '@kanaries/ml';
const trainX = [
    [0.12, 0.2, ..., 0.2],
    [0.21, 0.3, ..., 0.2],
    ...
];
const trainY = [0, 1, ..., 1];
const knn = new Neighbors.KNearstNeighbors(3, 'distance', '2-norm');
knn.fit(trainX, trainY);

const testX = [
    [0.52, 0.72, ..., 0.24],
    [0.11, 0.98, ..., 0.32],
    ...
];
knn.predict(testX)
// [2, 1, ..., 0]
```

## Development

```sh
# Install dependencies
yarn

# Run tests
yarn test

# Build the library
yarn build

# Start the development server for examples
yarn dev
```