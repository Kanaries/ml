import { createRandomGenerator } from './random';

export function std<T = any> (arr: T[], size: number, rng: () => number = Math.random): T[] {
    const n = arr.length;
    const k = Math.floor(size);
    if (n === 0 || k <= 0) {
        return [];
    }
    if (k >= n) {
        return arr.slice();
    }

    // Reservoir sampling (sampling without replacement).
    const sample = arr.slice(0, k);
    for (let i = k; i < n; i++) {
        const j = Math.floor(rng() * (i + 1));
        if (j < k) {
            sample[j] = arr[i];
        }
    }
    return sample;
}

export interface TrainTestSplitOptions {
    /** Fraction in (0, 1) or absolute count; defaults to 0.25 when trainSize is also unset. */
    testSize?: number;
    /** Fraction in (0, 1) or absolute count; defaults to the complement of testSize. */
    trainSize?: number;
    shuffle?: boolean;
    randomState?: number;
    /** Class labels; when given, the split preserves per-class proportions (requires shuffle). */
    stratify?: any[];
}

export interface TrainTestSplitResult<X = any, Y = any> {
    XTrain: X[];
    XTest: X[];
    yTrain?: Y[];
    yTest?: Y[];
}

/**
 * sklearn-style train/test sizing: float testSize rounds up, float trainSize
 * rounds down; whichever is missing takes the complement of the other.
 */
function resolveSplitCounts(sampleCount: number, testSize?: number, trainSize?: number): { testCount: number; trainCount: number } {
    if (testSize === undefined && trainSize === undefined) {
        testSize = 0.25;
    }

    let testCount: number | undefined;
    if (testSize !== undefined) {
        if (!Number.isFinite(testSize) || testSize <= 0) {
            throw new Error('testSize must be a positive finite number');
        }
        testCount = testSize < 1 ? Math.ceil(sampleCount * testSize) : Math.floor(testSize);
    }
    let trainCount: number | undefined;
    if (trainSize !== undefined) {
        if (!Number.isFinite(trainSize) || trainSize <= 0) {
            throw new Error('trainSize must be a positive finite number');
        }
        trainCount = trainSize < 1 ? Math.floor(sampleCount * trainSize) : Math.floor(trainSize);
    }

    if (trainCount === undefined) {
        trainCount = sampleCount - testCount!;
        if (testCount! < 1 || trainCount < 1) {
            throw new Error('testSize must produce at least 1 test sample and leave at least 1 train sample');
        }
    }
    if (testCount === undefined) {
        testCount = sampleCount - trainCount;
    }
    if (testCount < 1 || trainCount < 1 || testCount + trainCount > sampleCount) {
        throw new Error(`train/test sizes are invalid: trainCount=${trainCount} + testCount=${testCount} ` +
            `must fit in ${sampleCount} samples with at least 1 sample each`);
    }
    return { testCount, trainCount };
}

/**
 * Split `total` across categories proportionally to `counts` (largest-remainder
 * rounding, deterministic ties by category order); never allocates more than a
 * category's count. Used by stratified train/test splitting.
 */
export function allocateProportionally(counts: number[], total: number): number[] {
    const sum = counts.reduce((acc, c) => acc + c, 0);
    if (total > sum) {
        throw new Error(`cannot allocate ${total} samples across categories holding only ${sum}`);
    }
    const raw = counts.map(c => (c / sum) * total);
    const alloc = raw.map(Math.floor);
    let remaining = total - alloc.reduce((acc, c) => acc + c, 0);
    const order = raw
        .map((r, i) => ({ frac: r - Math.floor(r), i }))
        .sort((a, b) => b.frac - a.frac || a.i - b.i);
    for (const { i } of order) {
        if (remaining === 0) break;
        if (alloc[i] < counts[i]) {
            alloc[i]++;
            remaining--;
        }
    }
    // Safety net: if fractional-part ordering was exhausted (all capped), fill greedily.
    for (let i = 0; remaining > 0 && i < counts.length; i++) {
        if (alloc[i] < counts[i]) {
            alloc[i]++;
            remaining--;
        }
    }
    return alloc;
}

function stratifiedIndices(
    stratify: any[],
    trainCount: number,
    testCount: number,
    random: () => number,
): { trainIndices: number[]; testIndices: number[] } {
    const labelToIndices = new Map<any, number[]>();
    for (let i = 0; i < stratify.length; i++) {
        if (!labelToIndices.has(stratify[i])) {
            labelToIndices.set(stratify[i], []);
        }
        labelToIndices.get(stratify[i])!.push(i);
    }
    const classIndices = Array.from(labelToIndices.values());
    const counts = classIndices.map(indices => indices.length);
    for (const count of counts) {
        if (count < 2) {
            throw new Error('The least populated class in stratify has only 1 member, which is too few (minimum 2)');
        }
    }
    if (trainCount < counts.length || testCount < counts.length) {
        throw new Error('Both train and test sets must contain at least one sample per class; ' +
            'increase testSize/trainSize or reduce the number of classes');
    }

    const trainAlloc = allocateProportionally(counts, trainCount);
    const testAlloc = allocateProportionally(counts.map((c, i) => c - trainAlloc[i]), testCount);

    const trainIndices: number[] = [];
    const testIndices: number[] = [];
    for (let c = 0; c < classIndices.length; c++) {
        const shuffled = shuffledIndices(classIndices[c].length, random).map(i => classIndices[c][i]);
        trainIndices.push(...shuffled.slice(0, trainAlloc[c]));
        testIndices.push(...shuffled.slice(trainAlloc[c], trainAlloc[c] + testAlloc[c]));
    }
    // shuffle so downstream consumers don't see class-blocked ordering
    const trainOrder = shuffledIndices(trainIndices.length, random);
    const testOrder = shuffledIndices(testIndices.length, random);
    return {
        trainIndices: trainOrder.map(i => trainIndices[i]),
        testIndices: testOrder.map(i => testIndices[i]),
    };
}

function shuffledIndices(size: number, random: () => number): number[] {
    const indices = Array.from({ length: size }, (_, i) => i);
    for (let i = size - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

export function trainTestSplit<X = any, Y = any>(
    X: X[],
    y?: Y[],
    options: TrainTestSplitOptions = {},
): TrainTestSplitResult<X, Y> {
    const sampleCount = X.length;
    if (sampleCount < 2) {
        throw new Error('X must contain at least 2 samples');
    }
    if (y && y.length !== sampleCount) {
        throw new Error('X and y must have the same length');
    }
    if (options.stratify && options.stratify.length !== sampleCount) {
        throw new Error('stratify must have the same length as X');
    }
    if (options.stratify && options.shuffle === false) {
        throw new Error('stratified train/test split requires shuffle');
    }

    const { testCount, trainCount } = resolveSplitCounts(sampleCount, options.testSize, options.trainSize);

    let trainIndices: number[];
    let testIndices: number[];
    if (options.stratify) {
        ({ trainIndices, testIndices } = stratifiedIndices(
            options.stratify, trainCount, testCount, createRandomGenerator(options.randomState)));
    } else {
        const indices = options.shuffle === false
            ? Array.from({ length: sampleCount }, (_, i) => i)
            : shuffledIndices(sampleCount, createRandomGenerator(options.randomState));
        trainIndices = indices.slice(0, trainCount);
        testIndices = indices.slice(trainCount, trainCount + testCount);
    }

    const XTrain = trainIndices.map(i => X[i]);
    const XTest = testIndices.map(i => X[i]);

    if (!y) {
        return { XTrain, XTest };
    }

    const yTrain = trainIndices.map(i => y[i]);
    const yTest = testIndices.map(i => y[i]);
    return { XTrain, XTest, yTrain, yTest };
}
