export function std<T = any> (arr: T[], size: number): T[] {
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
        const j = Math.floor(Math.random() * (i + 1));
        if (j < k) {
            sample[j] = arr[i];
        }
    }
    return sample;
}

export interface TrainTestSplitOptions {
    testSize?: number;
    shuffle?: boolean;
    randomState?: number;
}

export interface TrainTestSplitResult<X = any, Y = any> {
    XTrain: X[];
    XTest: X[];
    yTrain?: Y[];
    yTest?: Y[];
}

function createRandomGenerator(seed?: number): () => number {
    if (seed === undefined) {
        return Math.random;
    }
    let state = Math.floor(seed) % 2147483647;
    if (state <= 0) {
        state += 2147483646;
    }
    return () => {
        state = (state * 16807) % 2147483647;
        return (state - 1) / 2147483646;
    };
}

function getTestCount(sampleCount: number, testSize?: number): number {
    if (testSize === undefined) {
        return Math.max(1, Math.floor(sampleCount * 0.25));
    }
    if (!Number.isFinite(testSize) || testSize <= 0) {
        throw new Error('testSize must be a positive finite number');
    }

    const count = testSize < 1 ? Math.floor(sampleCount * testSize) : Math.floor(testSize);
    if (count < 1 || count >= sampleCount) {
        throw new Error('testSize must produce at least 1 test sample and leave at least 1 train sample');
    }
    return count;
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

    const testCount = getTestCount(sampleCount, options.testSize);
    const trainCount = sampleCount - testCount;

    const indices = options.shuffle === false
        ? Array.from({ length: sampleCount }, (_, i) => i)
        : shuffledIndices(sampleCount, createRandomGenerator(options.randomState));

    const trainIndices = indices.slice(0, trainCount);
    const testIndices = indices.slice(trainCount);

    const XTrain = trainIndices.map(i => X[i]);
    const XTest = testIndices.map(i => X[i]);

    if (!y) {
        return { XTrain, XTest };
    }

    const yTrain = trainIndices.map(i => y[i]);
    const yTest = testIndices.map(i => y[i]);
    return { XTrain, XTest, yTrain, yTest };
}
