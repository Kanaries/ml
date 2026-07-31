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
