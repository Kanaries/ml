import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.resolve(TEST_DIR, '..', '..');

const clusterPages = [
  { file: 'index.mdx', pattern: /<ClusteringComparison \/>/ },
  { file: 'kmeans.mdx', pattern: /<KMeansPlayground \/>/ },
  { file: 'dbscan.mdx', pattern: /<ClusteringPlayground algorithm="dbscan" \/>/ },
  { file: 'hdbscan.mdx', pattern: /<ClusteringPlayground algorithm="hdbscan" \/>/ },
  { file: 'meanShift.mdx', pattern: /<ClusteringPlayground algorithm="meanShift" \/>/ },
  { file: 'optics.mdx', pattern: /<ClusteringPlayground algorithm="optics" \/>/ },
  { file: 'kmeansPlusPlus.mdx', pattern: /<ClusteringPlayground algorithm="kmeansPlusPlus" \/>/ },
  { file: 'advancedClustering.mdx', pattern: /<ClusteringPlayground algorithm="advanced" \/>/ },
];

const regressionPages = [
  ['linear/linearRegression.mdx', 'linear'],
  ['linear/polynomialRegression.mdx', 'polynomial'],
  ['linear/ridgeRegression.mdx', 'ridge'],
  ['linear/lassoRegression.mdx', 'lasso'],
  ['linear/elasticNet.mdx', 'elasticNet'],
  ['linear/robustRegressors.mdx', 'robust'],
  ['linear/bayesianRegressors.mdx', 'bayesian'],
  ['linear/generalizedLinearModels.mdx', 'glm'],
  ['ensemble/randomForestRegressor.mdx', 'randomForest'],
  ['ensemble/baggingRegressor.mdx', 'bagging'],
  ['ensemble/extraTreesRegressor.mdx', 'extraTrees'],
  ['ensemble/adaboostRegressor.mdx', 'adaBoost'],
  ['ensemble/gradientBoostingRegressor.mdx', 'gradientBoosting'],
  ['ensemble/xgboostRegressor.mdx', 'xgboost'],
  ['neighbors/kneighborsRegressor.mdx', 'knn'],
  ['neighbors/radiusNeighborsRegressor.mdx', 'radiusNeighbors'],
  ['svm/LinearSVR.mdx', 'linearSvr'],
  ['kernel/index.mdx', 'kernelRidge'],
  ['cross_decomposition/index.mdx', 'pls'],
  ['compose/index.mdx', 'transformedTarget'],
  ['multioutput/regressorChain.mdx', 'regressorChain'],
];

test('every clustering documentation page embeds an online interactive module', async () => {
  for (const page of clusterPages) {
    const source = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/apis/clusters', page.file), 'utf8');
    assert.match(source, page.pattern, `${page.file} is missing its clustering interaction`);
  }
});

test('every tree documentation page embeds the shared live tree module', async () => {
  for (const file of [
    'index.mdx',
    'decisionTreeClassifier.mdx',
    'decisionTreeRegressor.mdx',
    'extraTreeClassifier.mdx',
    'extraTreeRegressor.mdx',
  ]) {
    const source = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/apis/tree', file), 'utf8');
    assert.match(source, /<DecisionTreePlayground task=/, `${file} is missing its tree interaction`);
  }
});

test('every regression-focused documentation page embeds a live regression module', async () => {
  for (const [file, algorithm] of regressionPages) {
    const source = await fs.readFile(path.join(DOCS_ROOT, 'content/docs/apis', file), 'utf8');
    assert.match(
      source,
      new RegExp(`<RegressionPlayground algorithm=["']${algorithm}["'] \\/>`),
      `${file} is missing its ${algorithm} regression interaction`,
    );
  }
});

test('the shared regression playground wires every documented preset to @kanaries/ml', async () => {
  const source = await fs.readFile(path.join(DOCS_ROOT, 'components/regressionPlayground.tsx'), 'utf8');
  for (const api of [
    'Linear.LinearRegression',
    'Linear.PolynomialRegression',
    'Linear.RidgeRegression',
    'Linear.LassoRegression',
    'Linear.ElasticNet',
    'Linear.HuberRegressor',
    'Linear.RANSACRegressor',
    'Linear.TheilSenRegressor',
    'Linear.QuantileRegressor',
    'Linear.BayesianRidge',
    'Linear.ARDRegression',
    'Linear.PoissonRegressor',
    'Linear.GammaRegressor',
    'Linear.TweedieRegressor',
    'Ensemble.RandomForestRegressor',
    'Ensemble.BaggingRegressor',
    'Ensemble.ExtraTreesRegressor',
    'Ensemble.AdaBoostRegressor',
    'Ensemble.GradientBoostingRegressor',
    'Ensemble.XGBoostRegressor',
    'Neighbors.KNeighborsRegressor',
    'Neighbors.RadiusNeighborsRegressor',
    'SVM.LinearSVR',
    'Kernel.KernelRidge',
    'CrossDecomposition.PLSRegression',
    'Compose.TransformedTargetRegressor',
    'MultiOutput.RegressorChain',
  ]) {
    assert.ok(source.includes(api), `${api} is not wired into the regression playground`);
  }
  assert.match(source, /onClick={handlePlotClick}/, 'regression plot should accept user-added observations');
  assert.match(source, /aria-label="Add an observation by coordinates"/, 'regression plot should provide a keyboard-accessible coordinate input');
  assert.match(source, /aria-live="polite"/, 'regression metrics should announce live model changes');
});

test('every regression playground estimator variant fits and predicts with the published docs dependency', async () => {
  const {
    Compose,
    CrossDecomposition,
    Ensemble,
    Kernel,
    Linear,
    MultiOutput,
    Neighbors,
    SVM,
  } = await import('@kanaries/ml');
  const values = Array.from({ length: 36 }, (_, index) => -2.5 + index * 5 / 35);
  const X = values.map(value => [value]);
  const plsX = values.map(value => [value, value * value / 3]);
  const y = values.map((value, index) => Math.sin(value * 1.3) + value * 0.2 + 3 + (index % 13 === 0 ? 0.7 : 0));
  const multiY = y.map((value, index) => [value, 0.62 * value + 0.34 * values[index]]);
  const cases = [
    ['LinearRegression', new Linear.LinearRegression(), X, y],
    ['PolynomialRegression', new Linear.PolynomialRegression({ degree: 3 }), X, y],
    ['RidgeRegression', new Linear.RidgeRegression({ alpha: 0.2 }), X, y],
    ['LassoRegression', new Linear.LassoRegression({ alpha: 0.2, maxIter: 800, tol: 1e-6 }), X, y],
    ['ElasticNet', new Linear.ElasticNet({ alpha: 0.2, l1Ratio: 0.5, maxIter: 800, tol: 1e-6 }), X, y],
    ['HuberRegressor', new Linear.HuberRegressor({ epsilon: 1.35, maxIter: 180 }), X, y],
    ['RANSACRegressor', new Linear.RANSACRegressor({ residualThreshold: 0.45, maxTrials: 60, randomState: 17 }), X, y],
    ['TheilSenRegressor', new Linear.TheilSenRegressor({ maxSubpopulation: 160, randomState: 17 }), X, y],
    ['QuantileRegressor', new Linear.QuantileRegressor({ quantile: 0.5, alpha: 0.02, maxIter: 900, tol: 1e-5 }), X, y],
    ['BayesianRidge', new Linear.BayesianRidge({ maxIter: 150 }), X, y],
    ['ARDRegression', new Linear.ARDRegression({ maxIter: 150 }), X, y],
    ['PoissonRegressor', new Linear.PoissonRegressor({ alpha: 0.2, maxIter: 120 }), X, y],
    ['GammaRegressor', new Linear.GammaRegressor({ alpha: 0.2, maxIter: 120 }), X, y],
    ['TweedieRegressor', new Linear.TweedieRegressor({ power: 1.5, alpha: 0.2, link: 'log', maxIter: 120 }), X, y],
    ['RandomForestRegressor', new Ensemble.RandomForestRegressor({ nEstimators: 25, maxDepth: 5, randomState: 17 }), X, y],
    ['BaggingRegressor', new Ensemble.BaggingRegressor({ nEstimators: 25, maxSamples: 0.82, randomState: 17 }), X, y],
    ['ExtraTreesRegressor', new Ensemble.ExtraTreesRegressor({ nEstimators: 25, max_depth: 5, randomState: 17 }), X, y],
    ['AdaBoostRegressor', new Ensemble.AdaBoostRegressor({ nEstimators: 25, learningRate: 0.7, randomState: 17 }), X, y],
    ['GradientBoostingRegressor', new Ensemble.GradientBoostingRegressor({ nEstimators: 25, learningRate: 0.08, maxDepth: 3, randomState: 17 }), X, y],
    ['XGBoostRegressor', new Ensemble.XGBoostRegressor({ nEstimators: 25, learningRate: 0.18, maxDepth: 3, randomState: 17 }), X, y],
    ['KNeighborsRegressor', new Neighbors.KNeighborsRegressor({ nNeighbors: 5, weights: 'distance' }), X, y],
    ['RadiusNeighborsRegressor', new Neighbors.RadiusNeighborsRegressor({ radius: 0.7, weights: 'distance' }), X, y],
    ['LinearSVR', new SVM.LinearSVR({ C: 1, epsilon: 0.08, maxIter: 500, randomState: 17 }), X, y],
    ['KernelRidge', new Kernel.KernelRidge({ kernel: 'rbf', gamma: 0.8, alpha: 0.18 }), X, y],
    ['PLSRegression', new CrossDecomposition.PLSRegression({ nComponents: 2 }), plsX, multiY],
    ['TransformedTargetRegressor', new Compose.TransformedTargetRegressor({ regressor: new Linear.RidgeRegression({ alpha: 0.08 }), func: 'log1p', inverseFunc: 'expm1' }), X, y],
    ['RegressorChain', new MultiOutput.RegressorChain({ estimator: new Linear.RidgeRegression({ alpha: 0.2 }), order: [0, 1] }), X, multiY],
  ];

  for (const [name, model, features, targets] of cases) {
    model.fit(features, targets);
    const prediction = model.predict(features.slice(0, 5));
    const values = Array.isArray(prediction[0]) ? prediction.flat() : prediction;
    assert.equal(values.length > 0, true, `${name} returned no predictions`);
    assert.equal(values.every(Number.isFinite), true, `${name} returned a non-finite prediction`);
  }
});

test('the shared clustering playground fits every documented algorithm with @kanaries/ml', async () => {
  const source = await fs.readFile(path.join(DOCS_ROOT, 'components/clusteringPlayground.tsx'), 'utf8');
  for (const api of [
    'Clusters.DBScan',
    'Clusters.HDBScan',
    'Clusters.MeanShift',
    'Clusters.OPTICS',
    'Clusters.kmeansPlusPlus',
    'Clusters.Birch',
    'Clusters.AffinityPropagation',
    'Clusters.BisectingKMeans',
  ]) {
    assert.ok(source.includes(api), `${api} is not wired into the interactive component`);
  }
  assert.match(source, /onClick={handlePlotClick}/, 'cluster plot should accept user-added observations');
  assert.match(source, /aria-label="Add an observation by coordinates"/, 'cluster plot should provide a keyboard-accessible coordinate input');
  assert.match(source, /aria-live="polite"/, 'cluster metrics should announce live model changes');
  assert.match(source, /function clusterColor\(label/, 'cluster colors should remain distinct beyond the fixed palette');
  assert.match(source, /'Seed regions'/, 'k-means++ output should distinguish seed regions from fitted clusters');
  assert.match(source, /createRandom\(parameters\.modelSeed\)/, 'randomized seeding should remain reproducible');
});
