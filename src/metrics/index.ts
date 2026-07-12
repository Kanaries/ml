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
    if (positives === 0 || negatives === 0) {
        throw new Error('ROC AUC is undefined when only one class is present in the labels');
    }

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

function binaryRocAuc(expected: number[], scores: number[], positiveLabel: number): number {
    const roc = rocCurve(expected, scores, positiveLabel);
    let area = 0;
    for (let i = 1; i < roc.fpr.length; i++) {
        const width = roc.fpr[i] - roc.fpr[i - 1];
        const height = (roc.tpr[i] + roc.tpr[i - 1]) / 2;
        area += width * height;
    }
    return area;
}

export interface RocAucOptions {
    /** Positive label for the binary case (default 1), like the old third positional argument. */
    positiveLabel?: number;
    /**
     * Multiclass strategy, required when `scores` is a probability matrix
     * (sklearn's default multi_class='raise' behaviour).
     */
    multiClass?: 'ovr' | 'ovo';
    /** Averaging strategy for the multiclass case (sklearn default 'macro'). */
    average?: 'macro' | 'weighted';
}

/**
 * Area under the ROC curve. Mirrors sklearn.metrics.roc_auc_score.
 * Argument order: (expected, scores) — ground truth first, then scores,
 * matching rocCurve/precisionRecallCurve in this module.
 *
 * Binary: `scores` is a 1-D array of decision scores/probabilities of the
 * positive class (third argument may be the positive label, default 1).
 * Multiclass: `scores` is an n-samples x n-classes probability matrix whose
 * columns follow the sorted unique labels of `expected`; the third argument
 * must then supply `multiClass: 'ovr' | 'ovo'`.
 */
export function rocAucScore(expected: number[], scores: number[], positiveLabel?: number): number;
export function rocAucScore(expected: number[], scores: number[][], options?: RocAucOptions): number;
export function rocAucScore(
    expected: number[],
    scores: number[] | number[][],
    positiveLabelOrOptions: number | RocAucOptions = {},
): number {
    const options: RocAucOptions =
        typeof positiveLabelOrOptions === 'number'
            ? { positiveLabel: positiveLabelOrOptions }
            : positiveLabelOrOptions;
    if (scores.length > 0 && Array.isArray(scores[0])) {
        return multiClassRocAuc(expected, scores as number[][], options);
    }
    const positiveLabel = options.positiveLabel === undefined ? 1 : options.positiveLabel;
    return binaryRocAuc(expected, scores as number[], positiveLabel);
}

function multiClassRocAuc(expected: number[], scores: number[][], options: RocAucOptions): number {
    const { multiClass, average = 'macro' } = options;
    if (multiClass !== 'ovr' && multiClass !== 'ovo') {
        throw new Error("multiClass must be 'ovr' or 'ovo' when scores is a probability matrix");
    }
    if (expected.length !== scores.length) {
        throw new Error('actual and expected must have the same length');
    }
    if (expected.length === 0) {
        throw new Error('actual and expected must be non-empty');
    }
    const labels = Array.from(new Set(expected)).sort((a, b) => a - b);
    if (labels.length < 2) {
        throw new Error('multiclass rocAucScore requires at least two classes in expected');
    }
    if (labels.length !== scores[0].length) {
        throw new Error('Number of classes in expected not equal to the number of columns in scores');
    }
    for (const row of scores) {
        const sum = row.reduce((acc, value) => acc + value, 0);
        if (Math.abs(sum - 1) > 1e-5) {
            throw new Error('Target scores need to be probabilities for multiclass rocAucScore (rows must sum to 1)');
        }
    }

    const n = expected.length;
    if (multiClass === 'ovr') {
        const aucs: number[] = [];
        const prevalences: number[] = [];
        labels.forEach((label, index) => {
            const binary = expected.map(value => (value === label ? 1 : 0));
            const columnScores = scores.map(row => row[index]);
            aucs.push(binaryRocAuc(binary, columnScores, 1));
            prevalences.push(binary.reduce((acc, value) => acc + value, 0) / n);
        });
        if (average === 'weighted') {
            const weightSum = prevalences.reduce((acc, value) => acc + value, 0);
            return aucs.reduce((acc, auc, index) => acc + auc * prevalences[index], 0) / weightSum;
        }
        return aucs.reduce((acc, value) => acc + value, 0) / aucs.length;
    }

    // 'ovo' (Hand & Till style): average over class pairs of the mean of the
    // two directed binary AUCs restricted to samples of the pair.
    const pairScores: number[] = [];
    const prevalences: number[] = [];
    for (let a = 0; a < labels.length; a++) {
        for (let b = a + 1; b < labels.length; b++) {
            const indices: number[] = [];
            for (let i = 0; i < n; i++) {
                if (expected[i] === labels[a] || expected[i] === labels[b]) {
                    indices.push(i);
                }
            }
            const aTrue = indices.map(i => (expected[i] === labels[a] ? 1 : 0));
            const bTrue = indices.map(i => (expected[i] === labels[b] ? 1 : 0));
            const aScores = indices.map(i => scores[i][a]);
            const bScores = indices.map(i => scores[i][b]);
            const aAuc = binaryRocAuc(aTrue, aScores, 1);
            const bAuc = binaryRocAuc(bTrue, bScores, 1);
            pairScores.push((aAuc + bAuc) / 2);
            prevalences.push(indices.length / n);
        }
    }
    if (average === 'weighted') {
        const weightSum = prevalences.reduce((acc, value) => acc + value, 0);
        return pairScores.reduce((acc, score, index) => acc + score * prevalences[index], 0) / weightSum;
    }
    return pairScores.reduce((acc, value) => acc + value, 0) / pairScores.length;
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

interface ContingencyTable {
    matrix: number[][];
    rowSums: number[];
    colSums: number[];
    n: number;
}

function contingencyTable(labelsTrue: number[], labelsPred: number[]): ContingencyTable {
    const trueLabels = Array.from(new Set(labelsTrue)).sort((a, b) => a - b);
    const predLabels = Array.from(new Set(labelsPred)).sort((a, b) => a - b);
    const trueIndex = new Map<number, number>();
    const predIndex = new Map<number, number>();
    trueLabels.forEach((label, index) => trueIndex.set(label, index));
    predLabels.forEach((label, index) => predIndex.set(label, index));

    const matrix = Array.from({ length: trueLabels.length }, () => new Array(predLabels.length).fill(0));
    for (let i = 0; i < labelsTrue.length; i++) {
        matrix[trueIndex.get(labelsTrue[i])!][predIndex.get(labelsPred[i])!] += 1;
    }
    const rowSums = matrix.map(row => row.reduce((sum, value) => sum + value, 0));
    const colSums = predLabels.map((_, colIndex) => matrix.reduce((sum, row) => sum + row[colIndex], 0));
    return { matrix, rowSums, colSums, n: labelsTrue.length };
}

interface PairCombCounts {
    sumComb: number;
    sumCombRows: number;
    sumCombCols: number;
    totalComb: number;
}

function pairCombCounts(table: ContingencyTable): PairCombCounts {
    const sumComb = table.matrix.reduce(
        (sum, row) => sum + row.reduce((rowSum, value) => rowSum + combinations2(value), 0),
        0,
    );
    const sumCombRows = table.rowSums.reduce((sum, value) => sum + combinations2(value), 0);
    const sumCombCols = table.colSums.reduce((sum, value) => sum + combinations2(value), 0);
    return { sumComb, sumCombRows, sumCombCols, totalComb: combinations2(table.n) };
}

export function adjustedRandScore(labelsTrue: number[], labelsPred: number[]): number {
    assertSameLength(labelsTrue, labelsPred);
    const { sumComb, sumCombRows, sumCombCols, totalComb } = pairCombCounts(
        contingencyTable(labelsTrue, labelsPred),
    );

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

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

function meanOf(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** Entropy (natural log) of a label assignment, like sklearn.metrics.cluster.entropy. */
function labelEntropy(labels: number[]): number {
    const counts = new Map<number, number>();
    labels.forEach(label => counts.set(label, (counts.get(label) || 0) + 1));
    const n = labels.length;
    let entropy = 0;
    counts.forEach(count => {
        entropy -= (count / n) * (Math.log(count) - Math.log(n));
    });
    return entropy;
}

/** Lanczos approximation of ln(Gamma(x)) for x > 0. */
function logGamma(x: number): number {
    const g = 7;
    const coefficients = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (x < 0.5) {
        return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    }
    const z = x - 1;
    let sum = coefficients[0];
    for (let i = 1; i < coefficients.length; i++) {
        sum += coefficients[i] / (z + i);
    }
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

// ---------------------------------------------------------------------------
// Classification metrics
// ---------------------------------------------------------------------------

export interface LogLossOptions {
    /** Probability clipping epsilon; 'auto' (default) uses Number.EPSILON like modern sklearn. */
    eps?: number | 'auto';
    /** If true (default) return the mean per-sample loss, otherwise the sum. */
    normalize?: boolean;
    /** Explicit class labels (required when expected contains a single class). */
    labels?: number[];
}

/**
 * Log loss / cross-entropy. Mirrors sklearn.metrics.log_loss.
 * Argument order: (expected, predictedProbabilities) — ground truth first,
 * matching this module's score-based metrics (rocCurve/rocAucScore).
 *
 * `predictedProbabilities` is either 1-D (binary: probability of the greater
 * label) or an n-samples x n-classes matrix whose columns follow the sorted
 * class labels. Rows must sum to 1 (sklearn >= 1.5 raises otherwise).
 */
export function logLoss(
    expected: number[],
    predictedProbabilities: number[] | number[][],
    options: LogLossOptions = {},
): number {
    if (expected.length !== predictedProbabilities.length) {
        throw new Error('actual and expected must have the same length');
    }
    if (expected.length === 0) {
        throw new Error('actual and expected must be non-empty');
    }
    const { eps = 'auto', normalize = true, labels } = options;
    const epsilon = eps === 'auto' ? Number.EPSILON : eps;

    const classLabels = labels
        ? Array.from(new Set(labels)).sort((a, b) => a - b)
        : Array.from(new Set(expected)).sort((a, b) => a - b);

    let probabilityMatrix: number[][];
    if (Array.isArray(predictedProbabilities[0])) {
        probabilityMatrix = predictedProbabilities as number[][];
    } else {
        if (classLabels.length === 1) {
            throw new Error(
                'expected contains only one label; provide the labels option so binary probabilities can be interpreted',
            );
        }
        if (classLabels.length > 2) {
            throw new Error('1-D probabilities are only supported for binary problems; pass a probability matrix');
        }
        probabilityMatrix = (predictedProbabilities as number[]).map(p => [1 - p, p]);
    }
    if (classLabels.length < 2) {
        throw new Error('logLoss requires at least two class labels; provide the labels option');
    }
    if (probabilityMatrix[0].length !== classLabels.length) {
        throw new Error(
            `Number of classes (${classLabels.length}) does not match the number of probability columns (${probabilityMatrix[0].length})`,
        );
    }

    const labelIndex = new Map<number, number>();
    classLabels.forEach((label, index) => labelIndex.set(label, index));

    const sumTolerance = Math.sqrt(Number.EPSILON);
    let total = 0;
    for (let i = 0; i < expected.length; i++) {
        const row = probabilityMatrix[i];
        const rowSum = row.reduce((sum, value) => sum + value, 0);
        if (Math.abs(rowSum - 1) > sumTolerance) {
            throw new Error('The predicted probabilities do not sum to one');
        }
        const index = labelIndex.get(expected[i]);
        if (index === undefined) {
            throw new Error(`expected contains label ${expected[i]} which is not in labels`);
        }
        const clipped = Math.min(Math.max(row[index], epsilon), 1 - epsilon);
        total -= Math.log(clipped);
    }
    return normalize ? total / expected.length : total;
}

export interface BalancedAccuracyOptions {
    /** If true, chance-adjust the score so random performance scores 0 (sklearn `adjusted`). */
    adjusted?: boolean;
}

/**
 * Balanced accuracy: macro-average of per-class recall. Mirrors
 * sklearn.metrics.balanced_accuracy_score (classes without true samples are
 * dropped from the average, as sklearn does).
 * Argument order: (actual, expected) — predictions first, ground truth second,
 * matching accuracyScore.
 */
export function balancedAccuracyScore(
    actual: number[],
    expected: number[],
    options: BalancedAccuracyOptions = {},
): number {
    assertSameLength(actual, expected);
    const { adjusted = false } = options;
    const recalls: number[] = [];
    for (const label of uniqueLabels(actual, expected)) {
        if (supportForLabel(expected, label) === 0) {
            continue;
        }
        recalls.push(recallForLabel(actual, expected, label));
    }
    let score = meanOf(recalls);
    if (adjusted) {
        const chance = 1 / recalls.length;
        score = (score - chance) / (1 - chance);
    }
    return score;
}

/**
 * Matthews correlation coefficient using the multiclass-safe covariance
 * formulation. Mirrors sklearn.metrics.matthews_corrcoef, including the
 * documented convention of returning 0 when the denominator is 0.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function matthewsCorrcoef(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    const matrix = confusionMatrix(actual, expected);
    const n = actual.length;
    let trace = 0;
    for (let i = 0; i < matrix.length; i++) {
        trace += matrix[i][i];
    }
    const trueCounts = matrix.map(row => row.reduce((sum, value) => sum + value, 0));
    const predCounts = matrix[0].map((_, col) => matrix.reduce((sum, row) => sum + row[col], 0));
    let dotTP = 0;
    let dotPP = 0;
    let dotTT = 0;
    for (let k = 0; k < matrix.length; k++) {
        dotTP += trueCounts[k] * predCounts[k];
        dotPP += predCounts[k] * predCounts[k];
        dotTT += trueCounts[k] * trueCounts[k];
    }
    const covYtYp = trace * n - dotTP;
    const covYpYp = n * n - dotPP;
    const covYtYt = n * n - dotTT;
    if (covYpYp === 0 || covYtYt === 0) {
        return 0;
    }
    return covYtYp / Math.sqrt(covYtYt * covYpYp);
}

export interface CohenKappaOptions {
    /** Weighting scheme for disagreements (sklearn `weights`); undefined = unweighted. */
    weights?: 'linear' | 'quadratic';
    /** Explicit label list (sklearn `labels`). */
    labels?: number[];
}

/**
 * Cohen's kappa between two annotators. Mirrors sklearn.metrics.cohen_kappa_score
 * (symmetric in its two label arrays).
 * Argument order: (y1, y2) — two labelings, like sklearn.
 */
export function cohenKappaScore(y1: number[], y2: number[], options: CohenKappaOptions = {}): number {
    assertSameLength(y1, y2);
    const labels = options.labels
        ? Array.from(new Set(options.labels)).sort((a, b) => a - b)
        : uniqueLabels(y1, y2);
    const matrix = confusionMatrix(y1, y2, labels);
    const k = labels.length;
    const n = matrix.reduce((sum, row) => sum + row.reduce((rowSum, value) => rowSum + value, 0), 0);
    const rowSums = matrix.map(row => row.reduce((sum, value) => sum + value, 0));
    const colSums = labels.map((_, col) => matrix.reduce((sum, row) => sum + row[col], 0));

    let observedDisagreement = 0;
    let expectedDisagreement = 0;
    for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
            let weight: number;
            if (options.weights === 'linear') {
                weight = Math.abs(i - j);
            } else if (options.weights === 'quadratic') {
                weight = (i - j) * (i - j);
            } else {
                weight = i === j ? 0 : 1;
            }
            observedDisagreement += weight * matrix[i][j];
            expectedDisagreement += weight * ((rowSums[i] * colSums[j]) / n);
        }
    }
    return 1 - observedDisagreement / expectedDisagreement;
}

/**
 * Average hinge loss for binary problems. Mirrors sklearn.metrics.hinge_loss
 * for the binary case: the greater of the two labels is treated as the
 * positive class (+1), the other as -1, and loss_i = max(0, 1 - y_i * d_i).
 * Argument order: (expected, decisions) — ground truth first, then decision
 * function values, matching the module's score-based metrics.
 */
export function hingeLoss(expected: number[], decisions: number[]): number {
    assertSameLength(expected, decisions);
    const labels = Array.from(new Set(expected)).sort((a, b) => a - b);
    if (labels.length !== 2) {
        throw new Error('hingeLoss supports binary problems with exactly two labels in expected');
    }
    const positive = labels[1];
    let total = 0;
    for (let i = 0; i < expected.length; i++) {
        const y = expected[i] === positive ? 1 : -1;
        total += Math.max(0, 1 - y * decisions[i]);
    }
    return total / expected.length;
}

export interface BrierScoreLossOptions {
    /** Label treated as the positive class (sklearn `pos_label`). */
    positiveLabel?: number;
}

/**
 * Brier score loss for binary problems: mean((p_i - 1{y_i == pos})^2).
 * Mirrors sklearn.metrics.brier_score_loss. When positiveLabel is omitted it
 * defaults to 1 provided the labels are within {-1, 0, 1} (sklearn raises
 * otherwise, and so do we).
 * Argument order: (expected, probabilities) — ground truth first, then the
 * predicted probability of the positive class.
 */
export function brierScoreLoss(
    expected: number[],
    probabilities: number[],
    options: BrierScoreLossOptions = {},
): number {
    assertSameLength(expected, probabilities);
    for (const p of probabilities) {
        if (p < 0 || p > 1) {
            throw new Error('probabilities must be within the range [0, 1]');
        }
    }
    const labels = Array.from(new Set(expected));
    if (labels.length > 2) {
        throw new Error('brierScoreLoss only supports binary targets');
    }
    let positiveLabel = options.positiveLabel;
    if (positiveLabel === undefined) {
        const inferable = labels.every(label => label === -1 || label === 0 || label === 1);
        if (!inferable) {
            throw new Error('positiveLabel could not be inferred; specify options.positiveLabel explicitly');
        }
        positiveLabel = 1;
    }
    let total = 0;
    for (let i = 0; i < expected.length; i++) {
        const target = expected[i] === positiveLabel ? 1 : 0;
        const diff = probabilities[i] - target;
        total += diff * diff;
    }
    return total / expected.length;
}

export interface ClassificationReportRow {
    precision: number;
    recall: number;
    f1Score: number;
    support: number;
}

export interface ClassificationReportResult {
    perClass: { [label: string]: ClassificationReportRow };
    accuracy: number;
    macroAvg: ClassificationReportRow;
    weightedAvg: ClassificationReportRow;
}

/**
 * Structured classification report matching the numbers produced by
 * sklearn.metrics.classification_report(..., output_dict=True): per-class
 * precision/recall/F1/support plus accuracy, macro and support-weighted
 * averages (avg rows carry the total support).
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function classificationReport(actual: number[], expected: number[]): ClassificationReportResult {
    assertSameLength(actual, expected);
    const labels = uniqueLabels(actual, expected);
    const n = expected.length;

    const perClass: { [label: string]: ClassificationReportRow } = {};
    const rows: ClassificationReportRow[] = [];
    for (const label of labels) {
        const precision = precisionForLabel(actual, expected, label);
        const recall = recallForLabel(actual, expected, label);
        const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
        const row = { precision, recall, f1Score: f1, support: supportForLabel(expected, label) };
        perClass[String(label)] = row;
        rows.push(row);
    }

    const macroAvg: ClassificationReportRow = {
        precision: meanOf(rows.map(row => row.precision)),
        recall: meanOf(rows.map(row => row.recall)),
        f1Score: meanOf(rows.map(row => row.f1Score)),
        support: n,
    };
    const weightedAvg: ClassificationReportRow = {
        precision: rows.reduce((sum, row) => sum + row.precision * row.support, 0) / n,
        recall: rows.reduce((sum, row) => sum + row.recall * row.support, 0) / n,
        f1Score: rows.reduce((sum, row) => sum + row.f1Score * row.support, 0) / n,
        support: n,
    };

    return {
        perClass,
        accuracy: accuracyScore(actual, expected),
        macroAvg,
        weightedAvg,
    };
}

// ---------------------------------------------------------------------------
// Regression metrics
// ---------------------------------------------------------------------------

/**
 * Mean absolute error. Mirrors sklearn.metrics.mean_absolute_error.
 * Argument order: (actual, expected) — predictions first, ground truth second,
 * matching meanSquaredError.
 */
export function meanAbsoluteError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    let sum = 0;
    for (let i = 0; i < actual.length; i++) {
        sum += Math.abs(expected[i] - actual[i]);
    }
    return sum / actual.length;
}

/**
 * Mean absolute percentage error with sklearn's epsilon clamping:
 * mean(|y_true - y_pred| / max(|y_true|, eps)) with eps = machine epsilon,
 * mirroring sklearn.metrics.mean_absolute_percentage_error.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function meanAbsolutePercentageError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    let sum = 0;
    for (let i = 0; i < actual.length; i++) {
        sum += Math.abs(expected[i] - actual[i]) / Math.max(Math.abs(expected[i]), Number.EPSILON);
    }
    return sum / actual.length;
}

/**
 * Median absolute error. Mirrors sklearn.metrics.median_absolute_error
 * (even-length medians are the mean of the two central values).
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function medianAbsoluteError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    const absoluteErrors = actual
        .map((value, index) => Math.abs(expected[index] - value))
        .sort((a, b) => a - b);
    const mid = Math.floor(absoluteErrors.length / 2);
    if (absoluteErrors.length % 2 === 0) {
        return (absoluteErrors[mid - 1] + absoluteErrors[mid]) / 2;
    }
    return absoluteErrors[mid];
}

/**
 * Mean squared logarithmic error: mean((log1p(y_true) - log1p(y_pred))^2).
 * Mirrors sklearn.metrics.mean_squared_log_error, including throwing when
 * either targets or predictions contain negative values.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function meanSquaredLogError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    let sum = 0;
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] < 0 || expected[i] < 0) {
            throw new Error('Mean Squared Logarithmic Error cannot be used when targets contain negative values');
        }
        const diff = Math.log1p(expected[i]) - Math.log1p(actual[i]);
        sum += diff * diff;
    }
    return sum / actual.length;
}

/**
 * Explained variance score: 1 - Var(y_true - y_pred) / Var(y_true).
 * Mirrors sklearn.metrics.explained_variance_score with force_finite=True:
 * when Var(y_true) is 0 the score is 1 for a perfect fit and 0 otherwise.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function explainedVarianceScore(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    const residuals = expected.map((value, index) => value - actual[index]);
    const residualMean = meanOf(residuals);
    const numerator = meanOf(residuals.map(value => (value - residualMean) ** 2));
    const expectedMean = meanOf(expected);
    const denominator = meanOf(expected.map(value => (value - expectedMean) ** 2));
    if (denominator === 0) {
        return numerator === 0 ? 1 : 0;
    }
    return 1 - numerator / denominator;
}

/**
 * Maximum residual error. Mirrors sklearn.metrics.max_error.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function maxError(actual: number[], expected: number[]): number {
    assertSameLength(actual, expected);
    let max = 0;
    for (let i = 0; i < actual.length; i++) {
        max = Math.max(max, Math.abs(expected[i] - actual[i]));
    }
    return max;
}

/**
 * Root mean squared error: sqrt(meanSquaredError). Mirrors
 * sklearn.metrics.root_mean_squared_error.
 * Argument order: (actual, expected) — predictions first, ground truth second.
 */
export function rootMeanSquaredError(actual: number[], expected: number[]): number {
    return Math.sqrt(meanSquaredError(actual, expected));
}

// ---------------------------------------------------------------------------
// Clustering metrics (X-based)
// ---------------------------------------------------------------------------

function pairwiseEuclidean(X: number[][]): number[][] {
    const n = X.length;
    const distances = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const d = Distance.euclidean(X[i], X[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    return distances;
}

function assertValidClusterCount(nLabels: number, nSamples: number): void {
    if (nLabels < 2 || nLabels > nSamples - 1) {
        throw new Error(`Number of labels is ${nLabels}. Valid values are 2 to n_samples - 1 (inclusive)`);
    }
}

export interface SilhouetteOptions {
    /** 'euclidean' (default) treats X as points; 'precomputed' treats X as a distance matrix. */
    metric?: 'euclidean' | 'precomputed';
}

/**
 * Per-sample silhouette coefficients. Mirrors sklearn.metrics.silhouette_samples
 * (singleton clusters score 0; 0/0 degenerate samples are mapped to 0).
 * Argument order: (X, labels).
 */
export function silhouetteSamples(X: number[][], labels: number[], options: SilhouetteOptions = {}): number[] {
    const { metric = 'euclidean' } = options;
    const n = labels.length;
    if (X.length !== n) {
        throw new Error('X and labels must have the same length');
    }
    if (n === 0) {
        throw new Error('X and labels must be non-empty');
    }
    let distances: number[][];
    if (metric === 'precomputed') {
        for (let i = 0; i < n; i++) {
            if (X[i].length !== n) {
                throw new Error('precomputed distance matrix must be square');
            }
            if (X[i][i] !== 0) {
                throw new Error('The precomputed distance matrix must contain zeros on its diagonal');
            }
        }
        distances = X;
    } else {
        distances = pairwiseEuclidean(X);
    }

    const clusterLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
    assertValidClusterCount(clusterLabels.length, n);
    const clusterIndex = new Map<number, number>();
    clusterLabels.forEach((label, index) => clusterIndex.set(label, index));
    const clusterSizes = new Array(clusterLabels.length).fill(0);
    labels.forEach(label => {
        clusterSizes[clusterIndex.get(label)!] += 1;
    });

    const samples: number[] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        const own = clusterIndex.get(labels[i])!;
        if (clusterSizes[own] === 1) {
            samples[i] = 0;
            continue;
        }
        const sums = new Array(clusterLabels.length).fill(0);
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            sums[clusterIndex.get(labels[j])!] += distances[i][j];
        }
        const a = sums[own] / (clusterSizes[own] - 1);
        let b = Number.POSITIVE_INFINITY;
        for (let c = 0; c < clusterLabels.length; c++) {
            if (c === own || clusterSizes[c] === 0) continue;
            b = Math.min(b, sums[c] / clusterSizes[c]);
        }
        const denominator = Math.max(a, b);
        samples[i] = denominator === 0 ? 0 : (b - a) / denominator;
    }
    return samples;
}

/**
 * Mean silhouette coefficient over all samples. Mirrors
 * sklearn.metrics.silhouette_score (euclidean or precomputed distances;
 * requires 2 <= nLabels <= nSamples - 1 and throws otherwise).
 * Argument order: (X, labels).
 */
export function silhouetteScore(X: number[][], labels: number[], options: SilhouetteOptions = {}): number {
    return meanOf(silhouetteSamples(X, labels, options));
}

/**
 * Davies-Bouldin index (lower is better, 0 is the minimum). Mirrors
 * sklearn.metrics.davies_bouldin_score with euclidean distances.
 * Argument order: (X, labels).
 */
export function daviesBouldinScore(X: number[][], labels: number[]): number {
    const n = labels.length;
    if (X.length !== n) {
        throw new Error('X and labels must have the same length');
    }
    if (n === 0) {
        throw new Error('X and labels must be non-empty');
    }
    const clusterLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
    assertValidClusterCount(clusterLabels.length, n);
    const k = clusterLabels.length;
    const dims = X[0].length;
    const clusterIndex = new Map<number, number>();
    clusterLabels.forEach((label, index) => clusterIndex.set(label, index));

    const centroids = Array.from({ length: k }, () => new Array(dims).fill(0));
    const sizes = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
        const c = clusterIndex.get(labels[i])!;
        sizes[c] += 1;
        for (let d = 0; d < dims; d++) {
            centroids[c][d] += X[i][d];
        }
    }
    for (let c = 0; c < k; c++) {
        for (let d = 0; d < dims; d++) {
            centroids[c][d] /= sizes[c];
        }
    }

    const intraDists = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
        const c = clusterIndex.get(labels[i])!;
        intraDists[c] += Distance.euclidean(X[i], centroids[c]);
    }
    for (let c = 0; c < k; c++) {
        intraDists[c] /= sizes[c];
    }

    let total = 0;
    for (let i = 0; i < k; i++) {
        let worst = 0;
        for (let j = 0; j < k; j++) {
            if (j === i) continue;
            const centroidDistance = Distance.euclidean(centroids[i], centroids[j]);
            // sklearn maps zero centroid distances to +inf, making the ratio 0
            const ratio = centroidDistance === 0 ? 0 : (intraDists[i] + intraDists[j]) / centroidDistance;
            worst = Math.max(worst, ratio);
        }
        total += worst;
    }
    return total / k;
}

/**
 * Calinski-Harabasz index (variance-ratio criterion, higher is better).
 * Mirrors sklearn.metrics.calinski_harabasz_score, including returning 1.0
 * when the within-cluster dispersion is 0.
 * Argument order: (X, labels).
 */
export function calinskiHarabaszScore(X: number[][], labels: number[]): number {
    const n = labels.length;
    if (X.length !== n) {
        throw new Error('X and labels must have the same length');
    }
    if (n === 0) {
        throw new Error('X and labels must be non-empty');
    }
    const clusterLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
    assertValidClusterCount(clusterLabels.length, n);
    const k = clusterLabels.length;
    const dims = X[0].length;
    const clusterIndex = new Map<number, number>();
    clusterLabels.forEach((label, index) => clusterIndex.set(label, index));

    const overallMean = new Array(dims).fill(0);
    for (let i = 0; i < n; i++) {
        for (let d = 0; d < dims; d++) {
            overallMean[d] += X[i][d];
        }
    }
    for (let d = 0; d < dims; d++) {
        overallMean[d] /= n;
    }

    const centroids = Array.from({ length: k }, () => new Array(dims).fill(0));
    const sizes = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
        const c = clusterIndex.get(labels[i])!;
        sizes[c] += 1;
        for (let d = 0; d < dims; d++) {
            centroids[c][d] += X[i][d];
        }
    }
    for (let c = 0; c < k; c++) {
        for (let d = 0; d < dims; d++) {
            centroids[c][d] /= sizes[c];
        }
    }

    let extraDispersion = 0;
    for (let c = 0; c < k; c++) {
        let squared = 0;
        for (let d = 0; d < dims; d++) {
            squared += (centroids[c][d] - overallMean[d]) ** 2;
        }
        extraDispersion += sizes[c] * squared;
    }
    let intraDispersion = 0;
    for (let i = 0; i < n; i++) {
        const c = clusterIndex.get(labels[i])!;
        for (let d = 0; d < dims; d++) {
            intraDispersion += (X[i][d] - centroids[c][d]) ** 2;
        }
    }

    if (intraDispersion === 0) {
        return 1;
    }
    return (extraDispersion * (n - k)) / (intraDispersion * (k - 1));
}

// ---------------------------------------------------------------------------
// Clustering metrics (label-based)
// ---------------------------------------------------------------------------

/**
 * Mutual information (natural log) between two clusterings. Mirrors
 * sklearn.metrics.mutual_info_score (clipped at 0 to absorb float error).
 * Argument order: (labelsTrue, labelsPred), matching adjustedRandScore.
 */
export function mutualInfoScore(labelsTrue: number[], labelsPred: number[]): number {
    assertSameLength(labelsTrue, labelsPred);
    const { matrix, rowSums, colSums, n } = contingencyTable(labelsTrue, labelsPred);
    let mi = 0;
    for (let i = 0; i < rowSums.length; i++) {
        for (let j = 0; j < colSums.length; j++) {
            const nij = matrix[i][j];
            if (nij === 0) continue;
            mi += (nij / n) * (Math.log(nij * n) - Math.log(rowSums[i] * colSums[j]));
        }
    }
    return Math.max(mi, 0);
}

function homogeneityCompletenessVMeasure(
    labelsTrue: number[],
    labelsPred: number[],
): { homogeneity: number; completeness: number; vMeasure: number } {
    assertSameLength(labelsTrue, labelsPred);
    const entropyTrue = labelEntropy(labelsTrue);
    const entropyPred = labelEntropy(labelsPred);
    const mi = mutualInfoScore(labelsTrue, labelsPred);
    const homogeneity = entropyTrue === 0 ? 1 : mi / entropyTrue;
    const completeness = entropyPred === 0 ? 1 : mi / entropyPred;
    const vMeasure =
        homogeneity + completeness === 0
            ? 0
            : (2 * homogeneity * completeness) / (homogeneity + completeness);
    return { homogeneity, completeness, vMeasure };
}

/**
 * Homogeneity: each cluster contains only members of a single class.
 * Mirrors sklearn.metrics.homogeneity_score.
 * Argument order: (labelsTrue, labelsPred).
 */
export function homogeneityScore(labelsTrue: number[], labelsPred: number[]): number {
    return homogeneityCompletenessVMeasure(labelsTrue, labelsPred).homogeneity;
}

/**
 * Completeness: all members of a class are assigned to the same cluster.
 * Mirrors sklearn.metrics.completeness_score.
 * Argument order: (labelsTrue, labelsPred).
 */
export function completenessScore(labelsTrue: number[], labelsPred: number[]): number {
    return homogeneityCompletenessVMeasure(labelsTrue, labelsPred).completeness;
}

/**
 * V-measure: harmonic mean of homogeneity and completeness (beta = 1).
 * Mirrors sklearn.metrics.v_measure_score.
 * Argument order: (labelsTrue, labelsPred).
 */
export function vMeasureScore(labelsTrue: number[], labelsPred: number[]): number {
    return homogeneityCompletenessVMeasure(labelsTrue, labelsPred).vMeasure;
}

export type MutualInfoAverageMethod = 'min' | 'geometric' | 'arithmetic' | 'max';

export interface NormalizedMutualInfoOptions {
    /** How to average the two entropies for normalization (sklearn default 'arithmetic'). */
    averageMethod?: MutualInfoAverageMethod;
}

function generalizedAverage(a: number, b: number, method: MutualInfoAverageMethod): number {
    switch (method) {
        case 'min':
            return Math.min(a, b);
        case 'geometric':
            return Math.sqrt(a * b);
        case 'arithmetic':
            return (a + b) / 2;
        case 'max':
            return Math.max(a, b);
        default:
            throw new Error(`Unknown average method ${method}`);
    }
}

/**
 * Normalized mutual information: MI / average(H(true), H(pred)).
 * Mirrors sklearn.metrics.normalized_mutual_info_score, including the special
 * cases: two single-cluster labelings score 1.0 and MI of exactly 0 scores 0.
 * Argument order: (labelsTrue, labelsPred).
 */
export function normalizedMutualInfoScore(
    labelsTrue: number[],
    labelsPred: number[],
    options: NormalizedMutualInfoOptions = {},
): number {
    assertSameLength(labelsTrue, labelsPred);
    const { averageMethod = 'arithmetic' } = options;
    const trueCount = new Set(labelsTrue).size;
    const predCount = new Set(labelsPred).size;
    if (trueCount === 1 && predCount === 1) {
        return 1;
    }
    const mi = mutualInfoScore(labelsTrue, labelsPred);
    if (mi === 0) {
        return 0;
    }
    const normalizer = generalizedAverage(labelEntropy(labelsTrue), labelEntropy(labelsPred), averageMethod);
    return mi / normalizer;
}

/** Expected mutual information under the permutation model (sklearn's EMI). */
function expectedMutualInformation(table: ContingencyTable): number {
    const { rowSums, colSums, n } = table;
    let emi = 0;
    for (const a of rowSums) {
        for (const b of colSums) {
            const start = Math.max(a + b - n, 1);
            const end = Math.min(a, b);
            for (let nij = start; nij <= end; nij++) {
                const term1 = nij / n;
                const term2 = Math.log(n * nij) - Math.log(a * b);
                const logTerm3 =
                    logGamma(a + 1) + logGamma(b + 1) + logGamma(n - a + 1) + logGamma(n - b + 1)
                    - logGamma(n + 1) - logGamma(nij + 1) - logGamma(a - nij + 1)
                    - logGamma(b - nij + 1) - logGamma(n - a - b + nij + 1);
                emi += term1 * term2 * Math.exp(logTerm3);
            }
        }
    }
    return emi;
}

/**
 * Adjusted mutual information: (MI - E[MI]) / (average(H_true, H_pred) - E[MI]).
 * Mirrors sklearn.metrics.adjusted_mutual_info_score with the arithmetic
 * average default and the same epsilon-guarded denominator.
 * Argument order: (labelsTrue, labelsPred).
 */
export function adjustedMutualInfoScore(
    labelsTrue: number[],
    labelsPred: number[],
    options: NormalizedMutualInfoOptions = {},
): number {
    assertSameLength(labelsTrue, labelsPred);
    const { averageMethod = 'arithmetic' } = options;
    const trueCount = new Set(labelsTrue).size;
    const predCount = new Set(labelsPred).size;
    if (trueCount === 1 && predCount === 1) {
        return 1;
    }
    const table = contingencyTable(labelsTrue, labelsPred);
    const mi = mutualInfoScore(labelsTrue, labelsPred);
    const emi = expectedMutualInformation(table);
    const normalizer = generalizedAverage(labelEntropy(labelsTrue), labelEntropy(labelsPred), averageMethod);
    let denominator = normalizer - emi;
    if (denominator < 0) {
        denominator = Math.min(denominator, -Number.EPSILON);
    } else {
        denominator = Math.max(denominator, Number.EPSILON);
    }
    return (mi - emi) / denominator;
}

/**
 * Fowlkes-Mallows index: geometric mean of pairwise precision and recall.
 * Mirrors sklearn.metrics.fowlkes_mallows_score (0 when there are no pairs
 * clustered together in both assignments).
 * Argument order: (labelsTrue, labelsPred).
 */
export function fowlkesMallowsScore(labelsTrue: number[], labelsPred: number[]): number {
    assertSameLength(labelsTrue, labelsPred);
    const { matrix, rowSums, colSums, n } = contingencyTable(labelsTrue, labelsPred);
    const sumSquares = matrix.reduce(
        (sum, row) => sum + row.reduce((rowSum, value) => rowSum + value * value, 0),
        0,
    );
    const tk = sumSquares - n;
    const pk = rowSums.reduce((sum, value) => sum + value * value, 0) - n;
    const qk = colSums.reduce((sum, value) => sum + value * value, 0) - n;
    if (tk === 0) {
        return 0;
    }
    return Math.sqrt(tk / pk) * Math.sqrt(tk / qk);
}

/**
 * (Unadjusted) Rand index: fraction of sample pairs on which the two
 * clusterings agree. Mirrors sklearn.metrics.rand_score.
 * Argument order: (labelsTrue, labelsPred).
 */
export function randScore(labelsTrue: number[], labelsPred: number[]): number {
    assertSameLength(labelsTrue, labelsPred);
    const { sumComb, sumCombRows, sumCombCols, totalComb } = pairCombCounts(
        contingencyTable(labelsTrue, labelsPred),
    );
    if (totalComb === 0) {
        return 1;
    }
    const agreements = totalComb + 2 * sumComb - sumCombRows - sumCombCols;
    return agreements / totalComb;
}

export { Distance };
