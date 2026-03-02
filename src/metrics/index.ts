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
    const precision = precisionScore(actual, expected, options);
    const recall = recallScore(actual, expected, options);
    if (precision + recall === 0) {
        return 0;
    }
    return (2 * precision * recall) / (precision + recall);
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

export { Distance };
