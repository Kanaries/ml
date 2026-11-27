export type PerClassMetrics = {
  label: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  support: number;
};

export type MatrixSummary = {
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroSpecificity: number;
  macroF1: number;
  weightedF1: number;
  mcc: number;
  kappa: number;
  perClass: PerClassMetrics[];
};

const safeDivide = (numerator: number, denominator: number) => denominator === 0 ? 0 : numerator / denominator;
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function summarizeConfusionMatrix(matrix: number[][], labels: number[]): MatrixSummary {
  const total = matrix.flat().reduce((sum, value) => sum + value, 0);
  const rowSums = matrix.map((row) => row.reduce((sum, value) => sum + value, 0));
  const colSums = labels.map((_, column) => matrix.reduce((sum, row) => sum + row[column], 0));
  const trace = labels.reduce((sum, _, index) => sum + matrix[index][index], 0);
  const perClass = labels.map((label, index) => {
    const tp = matrix[index][index];
    const fn = rowSums[index] - tp;
    const fp = colSums[index] - tp;
    const tn = total - tp - fn - fp;
    const precision = safeDivide(tp, tp + fp);
    const recall = safeDivide(tp, tp + fn);
    const specificity = safeDivide(tn, tn + fp);
    return {
      label,
      precision,
      recall,
      specificity,
      f1: safeDivide(2 * precision * recall, precision + recall),
      support: rowSums[index],
    };
  });

  const observedAgreement = safeDivide(trace, total);
  const expectedAgreement = rowSums.reduce((sum, rowTotal, index) => sum + rowTotal * colSums[index], 0) / (total * total);
  const covTruePred = trace * total - rowSums.reduce((sum, rowTotal, index) => sum + rowTotal * colSums[index], 0);
  const covPred = total * total - colSums.reduce((sum, value) => sum + value * value, 0);
  const covTrue = total * total - rowSums.reduce((sum, value) => sum + value * value, 0);

  return {
    accuracy: observedAgreement,
    macroPrecision: mean(perClass.map((row) => row.precision)),
    macroRecall: mean(perClass.map((row) => row.recall)),
    macroSpecificity: mean(perClass.map((row) => row.specificity)),
    macroF1: mean(perClass.map((row) => row.f1)),
    weightedF1: perClass.reduce((sum, row) => sum + row.f1 * row.support, 0) / total,
    mcc: covPred === 0 || covTrue === 0 ? 0 : covTruePred / Math.sqrt(covPred * covTrue),
    kappa: expectedAgreement === 1 ? 0 : (observedAgreement - expectedAgreement) / (1 - expectedAgreement),
    perClass,
  };
}

export function binaryLogLoss(yTrue: number[], probabilities: number[], positiveLabel: number) {
  const epsilon = Number.EPSILON;
  return yTrue.reduce((sum, label, index) => {
    const target = label === positiveLabel ? 1 : 0;
    const probability = Math.min(1 - epsilon, Math.max(epsilon, probabilities[index]));
    return sum - target * Math.log(probability) - (1 - target) * Math.log(1 - probability);
  }, 0) / yTrue.length;
}

export type ThresholdEvaluation = {
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  fpr: number;
  specificity: number;
  f1: number;
  accuracy: number;
  balancedAccuracy: number;
  youdenJ: number;
};

/**
 * Average Precision matching `sklearn.metrics.average_precision_score`.
 * Accepts `precision` and `recall` in ascending-threshold order as returned by
 * `Metrics.precisionRecallCurve`, appending the standard (precision=1, recall=0) sentinel.
 */
export function computeAveragePrecision(precision: number[], recall: number[]): number {
  if (precision.length === 0 || precision.length !== recall.length) return 0;
  const p = [...precision, 1];
  const r = [...recall, 0];
  let ap = 0;
  for (let i = 0; i < precision.length; i++) {
    ap += (r[i] - r[i + 1]) * p[i];
  }
  return ap;
}

export function brierScore(yTrue: number[], yScores: number[], positiveLabel = 1): number {
  if (yTrue.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const target = yTrue[i] === positiveLabel ? 1 : 0;
    const diff = yScores[i] - target;
    sum += diff * diff;
  }
  return sum / yTrue.length;
}

export function evaluateThreshold(
  yTrue: number[],
  yScores: number[],
  threshold: number,
  positiveLabel = 1,
): ThresholdEvaluation {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const isPos = yTrue[i] === positiveLabel;
    const predPos = yScores[i] >= threshold;
    if (predPos && isPos) tp++;
    else if (predPos && !isPos) fp++;
    else if (!predPos && isPos) fn++;
    else tn++;
  }
  const precision = safeDivide(tp, tp + fp);
  const recall = safeDivide(tp, tp + fn);
  const fpr = safeDivide(fp, fp + tn);
  const specificity = safeDivide(tn, tn + fp);
  const f1 = safeDivide(2 * precision * recall, precision + recall);
  const accuracy = safeDivide(tp + tn, yTrue.length);
  const balancedAccuracy = (recall + specificity) / 2;
  const youdenJ = recall - fpr;
  return { threshold, tp, fp, fn, tn, precision, recall, fpr, specificity, f1, accuracy, balancedAccuracy, youdenJ };
}

export function findOptimalThresholds(
  yTrue: number[],
  yScores: number[],
  positiveLabel = 1,
): { bestYouden: ThresholdEvaluation; bestF1: ThresholdEvaluation } {
  const uniqueScores = Array.from(new Set(yScores)).sort((a, b) => a - b);
  let bestYouden = evaluateThreshold(yTrue, yScores, uniqueScores[0], positiveLabel);
  let bestF1 = bestYouden;
  for (const t of uniqueScores) {
    const ev = evaluateThreshold(yTrue, yScores, t, positiveLabel);
    if (ev.youdenJ > bestYouden.youdenJ) bestYouden = ev;
    if (ev.f1 > bestF1.f1) bestF1 = ev;
  }
  return { bestYouden, bestF1 };
}

export type RegressionMetricsSummary = {
  r2: number;
  adjustedR2: number;
  mse: number;
  rmse: number;
  mae: number;
  residuals: number[];
};

export function computeRegressionMetrics(
  yTrue: number[],
  yPred: number[],
  nFeatures: number,
): RegressionMetricsSummary {
  const n = yTrue.length;
  const residuals = yTrue.map((y, i) => y - yPred[i]);
  const yMean = mean(yTrue);
  let ssRes = 0;
  let ssTot = 0;
  let absSum = 0;
  for (let i = 0; i < n; i++) {
    ssRes += residuals[i] * residuals[i];
    const dy = yTrue[i] - yMean;
    ssTot += dy * dy;
    absSum += Math.abs(residuals[i]);
  }
  const mse = ssRes / n;
  const rmse = Math.sqrt(mse);
  const mae = absSum / n;
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  const adjustedR2 = n > nFeatures + 1 ? 1 - ((1 - r2) * (n - 1)) / (n - nFeatures - 1) : r2;
  return { r2, adjustedR2, mse, rmse, mae, residuals };
}

export function standardizeMatrix(X: number[][]): {
  scaled: number[][];
  means: number[];
  stds: number[];
} {
  const n = X.length;
  const p = X[0].length;
  const means = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < p; j++) means[j] /= n;

  const stds = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      const d = X[i][j] - means[j];
      stds[j] += d * d;
    }
  }
  for (let j = 0; j < p; j++) {
    const s = Math.sqrt(stds[j] / n);
    stds[j] = s === 0 ? 1 : s;
  }

  const scaled = X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
  return { scaled, means, stds };
}

export function unscaleLinearWeights(
  scaledCoef: number[],
  scaledIntercept: number,
  means: number[],
  stds: number[],
): { coef: number[]; intercept: number } {
  const coef = scaledCoef.map((w, j) => w / stds[j]);
  let shift = 0;
  for (let j = 0; j < scaledCoef.length; j++) {
    shift += (scaledCoef[j] * means[j]) / stds[j];
  }
  return { coef, intercept: scaledIntercept - shift };
}
