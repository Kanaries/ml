import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
import { Linear, Metrics } from '@kanaries/ml';

async function loadCalculatorMetrics() {
  const sourceUrl = new URL('../../components/tools/calculatorMetrics.ts', import.meta.url);
  const source = await readFile(sourceUrl, 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`);
}

test('confusion-matrix calculator metrics match sklearn reference values within 1e-6', async () => {
  const { summarizeConfusionMatrix } = await loadCalculatorMetrics();
  const yTrue = [0, 2, 2, 1, 1, 0];
  const yPred = [0, 1, 2, 2, 1, 0];
  const labels = [0, 1, 2];
  const matrix = Metrics.confusionMatrix(yPred, yTrue, labels);
  const summary = summarizeConfusionMatrix(matrix, labels);

  // sklearn.metrics.classification_report(..., output_dict=True),
  // matthews_corrcoef(...), and cohen_kappa_score(...).
  assert.deepEqual(matrix, [[2, 0, 0], [0, 1, 1], [0, 1, 1]]);
  assert.ok(Math.abs(summary.accuracy - 2 / 3) < 1e-6);
  assert.ok(Math.abs(summary.macroPrecision - 2 / 3) < 1e-6);
  assert.ok(Math.abs(summary.macroRecall - 2 / 3) < 1e-6);
  assert.ok(Math.abs(summary.macroF1 - 2 / 3) < 1e-6);
  assert.ok(Math.abs(summary.weightedF1 - 2 / 3) < 1e-6);
  assert.ok(Math.abs(summary.mcc - 0.5) < 1e-6);
  assert.ok(Math.abs(summary.kappa - 0.5) < 1e-6);
});

test('calculator binary log loss matches sklearn docs reference within 1e-6', async () => {
  const { binaryLogLoss } = await loadCalculatorMetrics();
  const yTrue = [1, 0, 0, 1];
  const probabilities = [0.9, 0.1, 0.2, 0.65];
  assert.ok(Math.abs(binaryLogLoss(yTrue, probabilities, 1) - 0.21616187468057912) < 1e-6);
});

test('logistic calculator coefficients match unregularized sklearn within 1e-6', () => {
  const X = [[0.5], [1], [1.5], [2], [2.4], [2.8], [3.2], [3.8], [4.2], [4.8], [5.4], [6]];
  const y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1];
  const mean = X.reduce((sum, row) => sum + row[0], 0) / X.length;
  const scale = Math.sqrt(X.reduce((sum, row) => sum + (row[0] - mean) ** 2, 0) / X.length);
  const scaledX = X.map((row) => [(row[0] - mean) / scale]);

  const model = new Linear.LogisticRegression({ learningRate: 0.2, maxIter: 30000 });
  model.fit(scaledX, y);
  const weights = model.coef;
  const bias = model.decisionFunction([[0]])[0];
  const coefficient = weights[0] / scale;
  const intercept = bias - (weights[0] * mean) / scale;

  // sklearn 1.7.2:
  // LogisticRegression(penalty=None, solver='lbfgs', max_iter=100000, tol=1e-14)
  assert.ok(Math.abs(coefficient - 3.07081117) < 1e-6);
  assert.ok(Math.abs(intercept - -7.99550043) < 1e-6);
});

async function loadClientUtils() {
  const sourceUrl = new URL('../../components/tools/clientUtils.ts', import.meta.url);
  const source = await readFile(sourceUrl, 'utf8');
  const output = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`);
}

test('ROC-AUC, Average Precision, Brier score, and optimal thresholds match sklearn within 1e-6', async () => {
  const { computeAveragePrecision, brierScore, findOptimalThresholds } = await loadCalculatorMetrics();
  const yTrue = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0];
  const yScores = [0.92, 0.81, 0.74, 0.68, 0.55, 0.49, 0.35, 0.22, 0.18, 0.08];

  const rocAuc = Metrics.rocAucScore(yTrue, yScores, 1);
  const pr = Metrics.precisionRecallCurve(yTrue, yScores, 1);
  const ap = computeAveragePrecision(pr.precision, pr.recall);
  const brier = brierScore(yTrue, yScores, 1);
  const { bestYouden, bestF1 } = findOptimalThresholds(yTrue, yScores, 1);

  assert.ok(Math.abs(rocAuc - 0.72) < 1e-6);
  assert.ok(Math.abs(ap - 0.7944444444444445) < 1e-6);
  assert.ok(Math.abs(brier - 0.21048) < 1e-6);
  assert.equal(bestYouden.threshold, 0.49);
  assert.ok(Math.abs(bestYouden.youdenJ - 0.4) < 1e-6);
  assert.equal(bestF1.threshold, 0.49);
});

test('Linear & standardized Ridge regression calculator helpers match sklearn within 1e-6', async () => {
  const { standardizeMatrix, unscaleLinearWeights, computeRegressionMetrics } = await loadCalculatorMetrics();
  const X = [[10, 2], [20, 4], [30, 5], [40, 8], [50, 11], [60, 12]];
  const y = [25, 48, 66, 92, 118, 134];

  const ols = new Linear.LinearRegression();
  ols.fit(X, y);
  const olsIntercept = ols.predict([[0, 0]])[0];
  const olsMetrics = computeRegressionMetrics(y, ols.predict(X), 2);
  assert.ok(Math.abs(ols.coef[0] - 1.3516129032258062) < 1e-6);
  assert.ok(Math.abs(ols.coef[1] - 4.161290322580645) < 1e-6);
  assert.ok(Math.abs(olsIntercept - 4.06451612903227) < 1e-6);
  assert.ok(Math.abs(olsMetrics.r2 - 0.999645981801252) < 1e-6);

  const { scaled, means, stds } = standardizeMatrix(X);
  const ridge = new Linear.RidgeRegression({ alpha: 2.0, fitIntercept: true });
  ridge.fit(scaled, y);
  const ridgeIntercept = ridge.predict([[0, 0]])[0];
  const unscaled = unscaleLinearWeights(ridge.coef, ridgeIntercept, means, stds);
  const ridgeMetrics = computeRegressionMetrics(y, ridge.predict(scaled), 2);
  assert.ok(Math.abs(unscaled.coef[0] - 0.967268578231967) < 1e-6);
  assert.ok(Math.abs(unscaled.coef[1] - 4.454162119257866) < 1e-6);
  assert.ok(Math.abs(unscaled.intercept - 15.466464927076089) < 1e-6);
  assert.ok(Math.abs(ridgeMetrics.r2 - 0.9788313698788033) < 1e-6);
});

test('shareable URL state round-trips and embed snippet preserves ?embed=1 and hash', async () => {
  const { encodeToolState, decodeToolState, buildEmbedSnippet } = await loadClientUtils();
  const state = { mode: 'ridge', alpha: 1.5, standardize: true, csv: 'x,y\n1,2\n3,4' };
  const encoded = encodeToolState(state);
  assert.ok(encoded && encoded.length > 0);
  assert.deepEqual(decodeToolState(encoded), state);

  const iframe = buildEmbedSnippet('/tools/roc-auc-calculator', 'ROC-AUC Calculator', `#s=${encoded}`);
  assert.match(iframe, /src="https:\/\/ml\.kanaries\.net\/tools\/roc-auc-calculator\?embed=1#s=/);
  assert.match(iframe, /<iframe\b/);
});
