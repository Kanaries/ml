import * as Distance from './distances';

function assertSameLength(actual: number[], expected: number[]) {
    if (actual.length !== expected.length) {
        throw new Error('actual and expected must have the same length');
    }
    if (actual.length === 0) {
        throw new Error('actual and expected must be non-empty');
    }
}

export function accuracyScore(actual: number[], expected: number[], normalize: boolean = true): number {
    assertSameLength(actual, expected);
    let score = 0;
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === expected[i]) score++;
    }
    if (normalize) {
        return score / actual.length;
    }
    return score;
}

export interface ClassificationMetricOptions {
    average?: 'binary' | 'macro';
    positiveLabel?: number;
}

export interface PrecisionRecallFscoreSupportResult {
    precision: number;
    recall: number;
    fScore: number;
    support: number[];
}

function uniqueLabels(actual: number[], expected: number[]): number[] {
    const labelSet = new Set<number>();
    actual.forEach(v => labelSet.add(v));
    expected.forEach(v => labelSet.add(v));
    return Array.from(labelSet.values()).sort((a, b) => a - b);
}

function precisionForLabel(actual: number[], expected: number[], label: number): number {
    let tp = 0;
    let fp = 0;
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === label && expected[i] === label) {
            tp++;
        } else if (actual[i] === label && expected[i] !== label) {
            fp++;
        }
    }
    return tp + fp === 0 ? 0 : tp / (tp + fp);
}

function recallForLabel(actual: number[], expected: number[], label: number): number {
    let tp = 0;
    let fn = 0;
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === label && expected[i] === label) {
            tp++;
        } else if (actual[i] !== label && expected[i] === label) {
            fn++;
        }
    }
    return tp + fn === 0 ? 0 : tp / (tp + fn);
}

export function precisionScore(
    actual: number[],
    expected: number[],
    options: ClassificationMetricOptions = {},
): number {
    assertSameLength(actual, expected);
    const { average = 'binary', positiveLabel = 1 } = options;
    if (average === 'binary') {
        return precisionForLabel(actual, expected, positiveLabel);
    }
    const labels = uniqueLabels(actual, expected);
    const scores = labels.map(label => precisionForLabel(actual, expected, label));
    return scores.reduce((acc, v) => acc + v, 0) / scores.length;
}

export function recallScore(
    actual: number[],
    expected: number[],
    options: ClassificationMetricOptions = {},
): number {
    assertSameLength(actual, expected);
    const { average = 'binary', positiveLabel = 1 } = options;
    if (average === 'binary') {
        return recallForLabel(actual, expected, positiveLabel);
    }
    const labels = uniqueLabels(actual, expected);
    const scores = labels.map(label => recallForLabel(actual, expected, label));
    return scores.reduce((acc, v) => acc + v, 0) / scores.length;
}

export function f1Score(
    actual: number[],
    expected: number[],
    options: ClassificationMetricOptions = {},
): number {
    assertSameLength(actual, expected);
    const { average = 'binary', positiveLabel = 1 } = options;
    if (average === 'binary') {
        const precision = precisionForLabel(actual, expected, positiveLabel);
        const recall = recallForLabel(actual, expected, positiveLabel);
        return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    }
    // macro: mean of per-class F1 (not the harmonic mean of macro P/R)
    const labels = uniqueLabels(actual, expected);
    const scores = labels.map(label => {
        const precision = precisionForLabel(actual, expected, label);
        const recall = recallForLabel(actual, expected, label);
        return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    });
    return scores.reduce((acc, v) => acc + v, 0) / scores.length;
}

function supportForLabel(expected: number[], label: number): number {
    let support = 0;
    for (const value of expected) {
        if (value === label) {
            support++;
        }
    }
    return support;
}

export function precisionRecallFscoreSupport(
    actual: number[],
    expected: number[],
    options: ClassificationMetricOptions = {},
): PrecisionRecallFscoreSupportResult {
    assertSameLength(actual, expected);
    const { average = 'binary', positiveLabel = 1 } = options;
    if (average === 'binary') {
        return {
            precision: precisionScore(actual, expected, { average, positiveLabel }),
            recall: recallScore(actual, expected, { average, positiveLabel }),
            fScore: f1Score(actual, expected, { average, positiveLabel }),
            support: [supportForLabel(expected, positiveLabel)],
        };
    }

    const labels = uniqueLabels(actual, expected);
    const precisionValues = labels.map(label => precisionForLabel(actual, expected, label));
    const recallValues = labels.map(label => recallForLabel(actual, expected, label));
    const fScores = precisionValues.map((precision, index) => {
        const recall = recallValues[index];
        return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    });

    return {
        precision: precisionValues.reduce((sum, value) => sum + value, 0) / labels.length,
        recall: recallValues.reduce((sum, value) => sum + value, 0) / labels.length,
        fScore: fScores.reduce((sum, value) => sum + value, 0) / labels.length,
        support: labels.map(label => supportForLabel(expected, label)),
    };
}

export function meanSquaredError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    let sum = 0;
    for (let i = 0; i < actual.length; i++) {
        const diff = expected[i] - actual[i];
        sum += diff * diff;
    }
    return sum / actual.length;
}

export function r2Score(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    const expectedMean = expected.reduce((acc, v) => acc + v, 0) / expected.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < actual.length; i++) {
        ssRes += (expected[i] - actual[i]) ** 2;
        ssTot += (expected[i] - expectedMean) ** 2;
    }
    if (ssTot === 0) {
        return ssRes === 0 ? 1 : 0;
    }
    return 1 - ssRes / ssTot;
}

export function confusionMatrix(actual: number[], expected: number[], labels?: number[]): number[][] {
    assertSameLength(actual, expected);
    const orderedLabels = labels || uniqueLabels(actual, expected);
    const labelIndex = new Map<number, number>();
    orderedLabels.forEach((label, index) => labelIndex.set(label, index));
    const matrix = Array.from({ length: orderedLabels.length }, () => new Array(orderedLabels.length).fill(0));

    for (let i = 0; i < actual.length; i++) {
        const row = labelIndex.get(expected[i]);
        const col = labelIndex.get(actual[i]);
        if (row !== undefined && col !== undefined) {
            matrix[row][col] += 1;
        }
    }
    return matrix;
}

function assertBinaryLabels(expected: number[], positiveLabel: number): void {
    const labels = new Set(expected);
    if (labels.size > 2 || !labels.has(positiveLabel)) {
        throw new Error('Binary metrics require expected labels with the configured positiveLabel');
    }
}

function sortScoresDescending(expected: number[], scores: number[]): Array<{ expected: number; score: number }> {
    return expected.map((value, index) => ({ expected: value, score: scores[index] }))
        .sort((a, b) => b.score - a.score);
}

export function rocCurve(
    expected: number[],
    scores: number[],
    positiveLabel: number = 1,
): { fpr: number[]; tpr: number[]; thresholds: number[] } {
    assertSameLength(expected, scores);
    assertBinaryLabels(expected, positiveLabel);

    const pairs = sortScoresDescending(expected, scores);
    const positives = expected.filter(value => value === positiveLabel).length;
    const negatives = expected.length - positives;

    const thresholds = [Number.POSITIVE_INFINITY];
    const tpr = [0];
    const fpr = [0];
    let truePositives = 0;
    let falsePositives = 0;

    for (let i = 0; i < pairs.length; i++) {
        if (pairs[i].expected === positiveLabel) {
            truePositives++;
        } else {
            falsePositives++;
        }
        const nextScore = i === pairs.length - 1 ? Number.NEGATIVE_INFINITY : pairs[i + 1].score;
        if (pairs[i].score !== nextScore) {
            thresholds.push(pairs[i].score);
            tpr.push(positives === 0 ? 0 : truePositives / positives);
            fpr.push(negatives === 0 ? 0 : falsePositives / negatives);
        }
    }

    return { fpr, tpr, thresholds };
}

export function rocAucScore(expected: number[], scores: number[], positiveLabel: number = 1): number {
    const roc = rocCurve(expected, scores, positiveLabel);
    let area = 0;
    for (let i = 1; i < roc.fpr.length; i++) {
        const width = roc.fpr[i] - roc.fpr[i - 1];
        const height = (roc.tpr[i] + roc.tpr[i - 1]) / 2;
        area += width * height;
    }
    return area;
}

export function precisionRecallCurve(
    expected: number[],
    scores: number[],
    positiveLabel: number = 1,
): { precision: number[]; recall: number[]; thresholds: number[] } {
    assertSameLength(expected, scores);
    assertBinaryLabels(expected, positiveLabel);

    const thresholds = Array.from(new Set(scores)).sort((a, b) => a - b);
    const positives = expected.filter(value => value === positiveLabel).length;
    const precision: number[] = [];
    const recall: number[] = [];

    for (const threshold of thresholds) {
        let tp = 0;
        let fp = 0;
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] >= threshold) {
                if (expected[i] === positiveLabel) {
                    tp++;
                } else {
                    fp++;
                }
            }
        }
        precision.push(tp + fp === 0 ? 1 : tp / (tp + fp));
        recall.push(positives === 0 ? 0 : tp / positives);
    }

    return { precision, recall, thresholds };
}

function combinations2(n: number): number {
    return n < 2 ? 0 : (n * (n - 1)) / 2;
}

export function adjustedRandScore(labelsTrue: number[], labelsPred: number[]): number {
    assertSameLength(labelsTrue, labelsPred);
    const trueLabels = Array.from(new Set(labelsTrue)).sort((a, b) => a - b);
    const predLabels = Array.from(new Set(labelsPred)).sort((a, b) => a - b);
    const trueIndex = new Map<number, number>();
    const predIndex = new Map<number, number>();
    trueLabels.forEach((label, index) => trueIndex.set(label, index));
    predLabels.forEach((label, index) => predIndex.set(label, index));

    const contingency = Array.from({ length: trueLabels.length }, () => new Array(predLabels.length).fill(0));
    for (let i = 0; i < labelsTrue.length; i++) {
        contingency[trueIndex.get(labelsTrue[i])!][predIndex.get(labelsPred[i])!] += 1;
    }

    const sumComb = contingency.reduce(
        (sum, row) => sum + row.reduce((rowSum, value) => rowSum + combinations2(value), 0),
        0,
    );
    const rowSums = contingency.map(row => row.reduce((sum, value) => sum + value, 0));
    const colSums = predLabels.map((_, colIndex) =>
        contingency.reduce((sum, row) => sum + row[colIndex], 0)
    );
    const sumCombRows = rowSums.reduce((sum, value) => sum + combinations2(value), 0);
    const sumCombCols = colSums.reduce((sum, value) => sum + combinations2(value), 0);
    const totalComb = combinations2(labelsTrue.length);

    if (totalComb === 0) {
        return 1;
    }

    const expectedIndex = (sumCombRows * sumCombCols) / totalComb;
    const maxIndex = 0.5 * (sumCombRows + sumCombCols);
    if (maxIndex === expectedIndex) {
        return 1;
    }
    return (sumComb - expectedIndex) / (maxIndex - expectedIndex);
}

export { Distance };
