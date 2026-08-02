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
