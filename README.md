# @kanaries/ml

![License](https://img.shields.io/github/license/kanaries/ml?color=%23FF7575)
![CI](https://github.com/kanaries/ml/actions/workflows/ci.yml/badge.svg)

**@kanaries/ml** is a JavaScript/TypeScript machine learning library with a scikit-learn-like API. It works in both browsers and Node.js environments, so Python users can move familiar estimator workflows into frontend and full-stack JavaScript projects.

## Features

- Classification and regression trees
- k-nearest neighbors
- Support vector machines
- Naive Bayes classifier
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

const knn = new Neighbors.KNearstNeighbors(3, 'distance', 'euclidean');
knn.fit(trainX, trainY);

const testX = [
    [0.52, 0.72, /* ... */ 0.24],
    [0.11, 0.98, /* ... */ 0.32],
];
const result = knn.predict(testX);
console.log(result);
```

## Python vs JavaScript / TypeScript Examples

If you already know scikit-learn, the fastest way to understand `@kanaries/ml` is to compare the same workflow side by side.

### LogisticRegression

<table>
<tr>
<th width="50%">Python (scikit-learn)</th>
<th width="50%">JavaScript / TypeScript (@kanaries/ml)</th>
</tr>
<tr>
<td>

```python
from sklearn.linear_model import LogisticRegression

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 1, 1, 0]

clf = LogisticRegression(max_iter=500, random_state=0)
clf.fit(X, y)
pred = clf.predict([[0.9, 0.8], [0.2, 0.1]])
```

</td>
<td>

```ts
import { Linear } from '@kanaries/ml';

const X = [[0, 0], [1, 1], [1, 0], [0, 1]];
const y = [0, 1, 1, 0];

const clf = new Linear.LogisticRegression({ learningRate: 0.1, maxIter: 800 });
clf.fit(X, y);
const pred = clf.predict([[0.9, 0.8], [0.2, 0.1]]);
```

</td>
</tr>
</table>

### KMeans

<table>
<tr>
<th width="50%">Python (scikit-learn)</th>
<th width="50%">JavaScript / TypeScript (@kanaries/ml)</th>
</tr>
<tr>
<td>

```python
from sklearn.cluster import KMeans

X = [[0, 0], [0.2, 0.1], [4, 4], [4.1, 4.2]]

model = KMeans(n_clusters=2, random_state=0, n_init='auto')
labels = model.fit_predict(X)
```

</td>
<td>

```ts
import { Clusters } from '@kanaries/ml';

const X = [[0, 0], [0.2, 0.1], [4, 4], [4.1, 4.2]];

const model = new Clusters.KMeans(2);
const labels = model.fitPredict(X);
```

</td>
</tr>
</table>

### DecisionTreeClassifier

<table>
<tr>
<th width="50%">Python (scikit-learn)</th>
<th width="50%">JavaScript / TypeScript (@kanaries/ml)</th>
</tr>
<tr>
<td>

```python
from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 1, 1, 0]

clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=0)
clf.fit(X, y)
pred = clf.predict([[0.9, 0.8], [0.1, 0.2]])
```

</td>
<td>

```ts
import { Tree } from '@kanaries/ml';

const X = [[0, 0], [1, 1], [1, 0], [0, 1]];
const y = [0, 1, 1, 0];

const clf = new Tree.DecisionTreeClassifier({ max_depth: 3, criterion: 'gini' });
clf.fit(X, y);
const pred = clf.predict([[0.9, 0.8], [0.1, 0.2]]);
```

</td>
</tr>
</table>

### IsolationForest

<table>
<tr>
<th width="50%">Python (scikit-learn)</th>
<th width="50%">JavaScript / TypeScript (@kanaries/ml)</th>
</tr>
<tr>
<td>

```python
from sklearn.ensemble import IsolationForest

X = [[0, 0], [0.1, 0.2], [0.2, 0.1], [8, 8]]

clf = IsolationForest(n_estimators=50, contamination=0.25, random_state=0)
clf.fit(X)
pred = clf.predict(X)
```

</td>
<td>

```ts
import { Ensemble } from '@kanaries/ml';

const X = [[0, 0], [0.1, 0.2], [0.2, 0.1], [8, 8]];

const clf = new Ensemble.IsolationForest(256, 50, 0.25);
clf.fit(X);
const pred = clf.predict(X);
```

</td>
</tr>
</table>

For side-by-side Python and JavaScript examples across the algorithm docs, see [the documentation site](https://ml.kanaries.net/docs/apis).

## Supported Algorithms

The library exposes several categories of algorithms:

- **Tree**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor`
- **Neighbors**: `KNearstNeighbors`, `BallTree`, `KDTree`
- **Linear Models**: `LinearRegression`, `LogisticRegression`, `PolynomialRegression`, `RidgeRegression`, `LassoRegression`
- **Support Vector Machines**: `SVC`, `NuSVC`, `LinearSVC`
- **Naive Bayes**: `BernoulliNB`, `CategoricalNB`
- **Clustering**: `KMeans`, `kmeansPlusPlus`, `DBScan`, `OPTICS`, `MeanShift`, `HDBScan`
- **Decomposition**: `PCA`
- **Manifold Learning**: `SpectralEmbedding`, `MDS`, `LocallyLinearEmbedding`, `TSNE`
 - **Ensemble**: `IsolationForest`, `AdaBoostClassifier`
- **Metrics**: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `meanSquaredError`, `r2Score`
- **Utilities**: `Sampling.trainTestSplit`, `Preprocessing.StandardScaler`, `Preprocessing.MinMaxScaler`, `ModelSelection.KFold`, `ModelSelection.crossValScore`, linear algebra helpers and math functions

## Advanced Features

### asyncMode

`asyncMode` runs a synchronous function in a worker (Web Worker or Node.js worker thread) and returns a `Promise`.

```ts
import { utils } from '@kanaries/ml';

const heavy = (x: number) => x * x;
const runAsync = utils.asyncMode(heavy);

const result = await runAsync(5);
```

### trainTestSplit

`utils.Sampling.trainTestSplit` splits samples into train/test sets and supports reproducible shuffling with `randomState`.

```ts
import { utils } from '@kanaries/ml';

const X = [[1], [2], [3], [4], [5]];
const y = [0, 0, 1, 1, 1];

const { XTrain, XTest, yTrain, yTest } = utils.Sampling.trainTestSplit(X, y, {
    testSize: 0.4,
    randomState: 42,
});
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
