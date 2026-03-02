import { accuracyScore } from '../metrics';

export interface KFoldProps {
    nSplits?: number;
    shuffle?: boolean;
    randomState?: number;
}

export interface FoldIndices {
    trainIndices: number[];
    testIndices: number[];
}

export interface EstimatorLike {
    fit(trainX: number[][], trainY: number[]): void;
    predict(testX: number[][]): number[];
    score?: (X: number[][], Y: number[]) => number;
}

export interface CrossValScoreOptions {
    cv?: number | KFold;
    scoring?: (actual: number[], expected: number[]) => number;
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

function shuffledIndices(size: number, random: () => number): number[] {
    const indices = Array.from({ length: size }, (_, i) => i);
    for (let i = size - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

export class KFold {
    private nSplits: number;
    private shuffle: boolean;
    private randomState?: number;

    constructor(props: KFoldProps = {}) {
        const { nSplits = 5, shuffle = false, randomState } = props;
        if (!Number.isInteger(nSplits) || nSplits < 2) {
            throw new Error('nSplits must be an integer >= 2');
        }
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.randomState = randomState;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        if (X.length < 2) {
            throw new Error('X must contain at least 2 samples');
        }
        if (y && y.length !== X.length) {
            throw new Error('X and y must have the same length');
        }
        if (this.nSplits > X.length) {
            throw new Error('nSplits cannot be greater than number of samples');
        }

        const random = createRandomGenerator(this.randomState);
        const indices = this.shuffle
            ? shuffledIndices(X.length, random)
            : Array.from({ length: X.length }, (_, i) => i);

        const foldSizes = new Array(this.nSplits).fill(Math.floor(X.length / this.nSplits));
        for (let i = 0; i < X.length % this.nSplits; i++) {
            foldSizes[i]++;
        }

        const folds: FoldIndices[] = [];
        let start = 0;
        for (let i = 0; i < this.nSplits; i++) {
            const end = start + foldSizes[i];
            const testIndices = indices.slice(start, end);
            const trainIndices = indices.slice(0, start).concat(indices.slice(end));
            folds.push({ trainIndices, testIndices });
            start = end;
        }
        return folds;
    }
}

export function crossValScore(
    estimatorFactory: () => EstimatorLike,
    X: number[][],
    y: number[],
    options: CrossValScoreOptions = {},
): number[] {
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }

    const splitter =
        typeof options.cv === 'number'
            ? new KFold({ nSplits: options.cv })
            : options.cv || new KFold();

    const folds = splitter.split(X, y);
    const scores: number[] = [];

    for (const fold of folds) {
        const trainX = fold.trainIndices.map(i => X[i]);
        const trainY = fold.trainIndices.map(i => y[i]);
        const testX = fold.testIndices.map(i => X[i]);
        const testY = fold.testIndices.map(i => y[i]);

        const estimator = estimatorFactory();
        estimator.fit(trainX, trainY);

        if (options.scoring) {
            const predY = estimator.predict(testX);
            scores.push(options.scoring(predY, testY));
            continue;
        }

        if (typeof estimator.score === 'function') {
            scores.push(estimator.score(testX, testY));
            continue;
        }

        const predY = estimator.predict(testX);
        scores.push(accuracyScore(predY, testY));
    }

    return scores;
}
