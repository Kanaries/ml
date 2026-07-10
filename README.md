# @kanaries/ml — Machine Learning in JavaScript & TypeScript

[![npm version](https://img.shields.io/npm/v/%40kanaries%2Fml)](https://www.npmjs.com/package/@kanaries/ml)
[![npm downloads](https://img.shields.io/npm/dm/%40kanaries%2Fml)](https://www.npmjs.com/package/@kanaries/ml)
![CI](https://github.com/kanaries/ml/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/kanaries/ml?color=%23FF7575)

**@kanaries/ml** is a machine learning library for JavaScript and TypeScript with a scikit-learn-style API. Train and run classification, regression, clustering, dimensionality reduction, and anomaly detection models directly in the browser or in Node.js — no Python service required. If you know scikit-learn, you already know most of this library: estimators follow the same `fit` / `predict` workflow, naming, and options wherever practical.

[Documentation](https://ml.kanaries.net/docs) · [API Reference](https://ml.kanaries.net/docs/apis) · [npm](https://www.npmjs.com/package/@kanaries/ml) · [Issues](https://github.com/Kanaries/ml/issues)

## Features

- **50+ estimators** across classification, regression, clustering, dimensionality reduction, manifold learning, anomaly detection, and semi-supervised learning
- **scikit-learn-style API** — `fit`, `predict`, `fitPredict`, transformers, and metrics that mirror the Python ecosystem
- **Gradient boosting and ensembles** — `XGBoostClassifier`/`XGBoostRegressor`, `GradientBoosting`, `AdaBoost` (multiclass via SAMME), `RandomForest`, `Bagging`, and `IsolationForest`
- **Model selection built in** — `KFold`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, and `crossValScore`
- **Evaluation metrics** — accuracy, precision/recall/F1, confusion matrix, ROC curve, ROC AUC, precision-recall curve, MSE, R², adjusted Rand index
- **Runs anywhere JavaScript runs** — browsers, Node.js, and edge runtimes, with Web Worker support via `asyncMode`
- **TypeScript-first** — written in TypeScript with full type definitions shipped
- **Zero runtime dependencies** — nothing else gets pulled into your bundle

## How it compares

@kanaries/ml focuses on **classical machine learning** — the scikit-learn side of ML — rather than deep learning:

| Library | Focus | Choose it when |
| --- | --- | --- |
| **@kanaries/ml** | Classical ML with a scikit-learn-style API | You work with tabular data — classification, regression, clustering, anomaly detection — and want it in JS/TS without a Python backend |
| **TensorFlow.js** | Deep learning, GPU-accelerated tensors | You need neural networks, computer vision, or NLP models in the browser |
| **ml.js** | Collection of standalone numeric/ML packages | You want individual algorithms as separate small packages |

## Installation

```sh
npm install @kanaries/ml
# or
yarn add @kanaries/ml
```

## Quick Start

```ts
import { Neighbors } from '@kanaries/ml';

const trainX = [
    [0.12, 0.2, /* ... */ 0.2],
    [0.21, 0.3, /* ... */ 0.2],
];
const trainY = [0, 1];

const knn = new Neighbors.KNearestNeighbors(3, 'distance', 'euclidean');
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

- **[Tree](https://ml.kanaries.net/docs/apis/tree)**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreeRegressor`
- **[Ensemble](https://ml.kanaries.net/docs/apis/ensemble)**: `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `XGBoostClassifier`, `XGBoostRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`, `BaggingClassifier`, `IsolationForest`
- **[Linear Models](https://ml.kanaries.net/docs/apis/linear)**: `LinearRegression`, `LogisticRegression`, `PolynomialRegression`, `Ridge`, `RidgeRegression`, `RidgeClassifier`, `Lasso`, `LassoRegression`, `ElasticNet`
- **[Support Vector Machines](https://ml.kanaries.net/docs/apis/svm)**: `SVC`, `NuSVC` (SMO dual solvers, one-vs-one multiclass, linear/rbf/poly/sigmoid kernels), `LinearSVC`, `LinearSVR`
- **[Neighbors](https://ml.kanaries.net/docs/apis/neighbors)**: `KNearestNeighbors`, `KNeighborsRegressor`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`, `NearestCentroid`, `BallTree`, `KDTree`
- **[Naive Bayes](https://ml.kanaries.net/docs/apis/bayes)**: `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB`
- **[Clustering](https://ml.kanaries.net/docs/apis/clusters)**: `KMeans`, `kmeansPlusPlus`, `DBScan`, `HDBScan`, `OPTICS`, `MeanShift`
- **[Decomposition](https://ml.kanaries.net/docs/apis/decomposition)**: `PCA`, `TruncatedSVD`, `SparsePCA`
- **[Manifold Learning](https://ml.kanaries.net/docs/apis/manifold)**: `TSNE`, `MDS`, `SpectralEmbedding`, `LocallyLinearEmbedding`
- **[Semi-Supervised](https://ml.kanaries.net/docs/apis/semi_supervised)**: `LabelPropagation`, `LabelSpreading`
- **[Neural Network](https://ml.kanaries.net/docs/apis/neural_network)**: `BernoulliRBM`
- **[Metrics](https://ml.kanaries.net/docs/apis/metrics)**: `accuracyScore`, `precisionScore`, `recallScore`, `f1Score`, `precisionRecallFscoreSupport`, `confusionMatrix`, `rocCurve`, `rocAucScore`, `precisionRecallCurve`, `meanSquaredError`, `r2Score`, `adjustedRandScore`
- **[Utilities](https://ml.kanaries.net/docs/apis/utils)**: `Sampling.trainTestSplit`, `Preprocessing.StandardScaler`, `Preprocessing.MinMaxScaler`, `Preprocessing.MaxAbsScaler`, `ModelSelection.KFold`, `ModelSelection.StratifiedKFold`, `ModelSelection.GridSearchCV`, `ModelSelection.RandomizedSearchCV`, `ModelSelection.crossValScore`, linear algebra helpers and math functions

`KNearstNeighbors` remains available as a deprecated compatibility alias of `KNearestNeighbors`.

## Advanced Features

### Model selection

Tune hyperparameters and validate models the same way you would in scikit-learn:

```ts
import { utils, Tree } from '@kanaries/ml';

const search = new utils.ModelSelection.GridSearchCV({
    estimatorFactory: (params) => new Tree.DecisionTreeClassifier(params),
    paramGrid: { max_depth: [2, 3, 5], criterion: ['gini', 'entropy'] },
    cv: 5,
});
search.fit(X, y);
console.log(search.bestParams, search.bestScore);
```

### asyncMode

`asyncMode` runs a synchronous function in a worker (Web Worker or Node.js worker thread) and returns a `Promise`, keeping UIs responsive during training:

```ts
import { utils } from '@kanaries/ml';

const heavy = (x: number) => x * x;
const runAsync = utils.asyncMode(heavy);

const result = await runAsync(5);
```

### trainTestSplit

`utils.Sampling.trainTestSplit` splits samples into train/test sets and supports reproducible shuffling with `randomState`:

```ts
import { utils } from '@kanaries/ml';

const X = [[1], [2], [3], [4], [5]];
const y = [0, 0, 1, 1, 1];

const { XTrain, XTest, yTrain, yTest } = utils.Sampling.trainTestSplit(X, y, {
    testSize: 0.4,
    randomState: 42,
});
```

## Documentation

Full guides, algorithm explanations, and API references live at **[ml.kanaries.net/docs](https://ml.kanaries.net/docs)**. Every algorithm page includes runnable JavaScript examples with their Python equivalents.

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

## License

[MIT](./LICENSE)
