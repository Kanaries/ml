import { accuracyScore } from '../metrics';
import { createRandomGenerator } from './random';

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
    cv?: number | SplitterLike;
    scoring?: (actual: number[], expected: number[]) => number;
}

export interface SplitterLike {
    split(X: any[], y?: any[]): FoldIndices[];
}

export interface SearchEstimatorFactory {
    (params: Record<string, any>): EstimatorLike;
}

export interface GridSearchCVProps {
    estimatorFactory: SearchEstimatorFactory;
    paramGrid: Record<string, any[]>;
    cv?: number | SplitterLike;
    scoring?: (actual: number[], expected: number[]) => number;
    refit?: boolean;
}

export interface RandomizedSearchCVProps {
    estimatorFactory: SearchEstimatorFactory;
    paramDistributions: Record<string, any[]>;
    nIter: number;
    cv?: number | SplitterLike;
    scoring?: (actual: number[], expected: number[]) => number;
    randomState?: number;
    refit?: boolean;
}

function shuffledIndices(size: number, random: () => number): number[] {
    const indices = Array.from({ length: size }, (_, i) => i);
    for (let i = size - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

function resolveSplitter(cv?: number | SplitterLike): SplitterLike {
    return typeof cv === 'number' ? new KFold({ nSplits: cv }) : (cv || new KFold());
}

function mean(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parameterCombinations(grid: Record<string, any[]>): Record<string, any>[] {
    const keys = Object.keys(grid);
    if (keys.length === 0) {
        throw new Error('parameter grid must contain at least one hyperparameter');
    }
    for (const key of keys) {
        const values = grid[key];
        if (!Array.isArray(values) || values.length === 0) {
            throw new Error(`parameter grid for ${key} must be a non-empty array`);
        }
    }

    const combinations: Record<string, any>[] = [];
    function build(index: number, current: Record<string, any>): void {
        if (index === keys.length) {
            combinations.push({ ...current });
            return;
        }
        const key = keys[index];
        for (const value of grid[key]) {
            current[key] = value;
            build(index + 1, current);
        }
    }
    build(0, {});
    return combinations;
}

function evaluateEstimator(
    estimatorFactory: SearchEstimatorFactory,
    params: Record<string, any>,
    X: number[][],
    y: number[],
    cv?: number | SplitterLike,
    scoring?: (actual: number[], expected: number[]) => number,
): number {
    const splitter = resolveSplitter(cv);
    const folds = splitter.split(X, y);
    const scores = folds.map(fold => {
        const trainX = fold.trainIndices.map(i => X[i]);
        const trainY = fold.trainIndices.map(i => y[i]);
        const testX = fold.testIndices.map(i => X[i]);
        const testY = fold.testIndices.map(i => y[i]);
        const estimator = estimatorFactory(params);
        estimator.fit(trainX, trainY);
        if (scoring) {
            return scoring(estimator.predict(testX), testY);
        }
        if (typeof estimator.score === 'function') {
            return estimator.score(testX, testY);
        }
        return accuracyScore(estimator.predict(testX), testY);
    });
    return mean(scores);
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

export class StratifiedKFold implements SplitterLike {
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
        if (!y) {
            throw new Error('StratifiedKFold requires y labels');
        }
        if (X.length < 2) {
            throw new Error('X must contain at least 2 samples');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        if (this.nSplits > X.length) {
            throw new Error('nSplits cannot be greater than number of samples');
        }

        const labelToIndices = new Map<any, number[]>();
        for (let i = 0; i < y.length; i++) {
            const label = y[i];
            if (!labelToIndices.has(label)) {
                labelToIndices.set(label, []);
            }
            labelToIndices.get(label)!.push(i);
        }
        for (const indices of labelToIndices.values()) {
            if (indices.length < this.nSplits) {
                throw new Error('Each class must have at least nSplits samples');
            }
        }

        const random = createRandomGenerator(this.randomState);
        const foldBuckets: number[][] = Array.from({ length: this.nSplits }, () => []);
        for (const indices of labelToIndices.values()) {
            const ordered = this.shuffle ? shuffledIndices(indices.length, random).map(i => indices[i]) : indices.slice();
            for (let i = 0; i < ordered.length; i++) {
                foldBuckets[i % this.nSplits].push(ordered[i]);
            }
        }

        return foldBuckets.map(testIndicesRaw => {
            const testSet = new Set(testIndicesRaw);
            const testIndices = testIndicesRaw.slice().sort((a, b) => a - b);
            const trainIndices = Array.from({ length: X.length }, (_, i) => i).filter(i => !testSet.has(i));
            return { trainIndices, testIndices };
        });
    }
}

export class GridSearchCV {
    private estimatorFactory: SearchEstimatorFactory;
    private paramGrid: Record<string, any[]>;
    private cv?: number | SplitterLike;
    private scoring?: (actual: number[], expected: number[]) => number;
    private refit: boolean;
    public bestParams: Record<string, any> | null;
    public bestScore: number;
    public bestEstimator: EstimatorLike | null;

    constructor(props: GridSearchCVProps) {
        this.estimatorFactory = props.estimatorFactory;
        this.paramGrid = props.paramGrid;
        this.cv = props.cv;
        this.scoring = props.scoring;
        this.refit = props.refit ?? true;
        this.bestParams = null;
        this.bestScore = Number.NEGATIVE_INFINITY;
        this.bestEstimator = null;
    }

    public fit(X: number[][], y: number[]): void {
        if (X.length === 0 || y.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }

        const combinations = parameterCombinations(this.paramGrid);
        this.bestParams = null;
        this.bestScore = Number.NEGATIVE_INFINITY;
        this.bestEstimator = null;

        for (const params of combinations) {
            const score = evaluateEstimator(this.estimatorFactory, params, X, y, this.cv, this.scoring);
            if (score > this.bestScore) {
                this.bestScore = score;
                this.bestParams = { ...params };
            }
        }

        if (this.refit && this.bestParams) {
            this.bestEstimator = this.estimatorFactory(this.bestParams);
            this.bestEstimator.fit(X, y);
        }
    }

    public predict(X: number[][]): number[] {
        if (!this.bestEstimator) {
            throw new Error('search must be fitted before calling predict');
        }
        return this.bestEstimator.predict(X);
    }

    public score(X: number[][], y: number[]): number {
        if (!this.bestEstimator) {
            throw new Error('search must be fitted before calling score');
        }
        if (this.scoring) {
            return this.scoring(this.bestEstimator.predict(X), y);
        }
        if (typeof this.bestEstimator.score === 'function') {
            return this.bestEstimator.score(X, y);
        }
        return accuracyScore(this.bestEstimator.predict(X), y);
    }
}

export class RandomizedSearchCV {
    private estimatorFactory: SearchEstimatorFactory;
    private paramDistributions: Record<string, any[]>;
    private nIter: number;
    private cv?: number | SplitterLike;
    private scoring?: (actual: number[], expected: number[]) => number;
    private randomState?: number;
    private refit: boolean;
    public bestParams: Record<string, any> | null;
    public bestScore: number;
    public bestEstimator: EstimatorLike | null;

    constructor(props: RandomizedSearchCVProps) {
        if (!Number.isInteger(props.nIter) || props.nIter <= 0) {
            throw new Error('nIter must be an integer > 0');
        }
        this.estimatorFactory = props.estimatorFactory;
        this.paramDistributions = props.paramDistributions;
        this.nIter = props.nIter;
        this.cv = props.cv;
        this.scoring = props.scoring;
        this.randomState = props.randomState;
        this.refit = props.refit ?? true;
        this.bestParams = null;
        this.bestScore = Number.NEGATIVE_INFINITY;
        this.bestEstimator = null;
    }

    public fit(X: number[][], y: number[]): void {
        if (X.length === 0 || y.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }

        const combinations = parameterCombinations(this.paramDistributions);
        const random = createRandomGenerator(this.randomState);
        const order = shuffledIndices(combinations.length, random).slice(0, Math.min(this.nIter, combinations.length));
        this.bestParams = null;
        this.bestScore = Number.NEGATIVE_INFINITY;
        this.bestEstimator = null;

        for (const index of order) {
            const params = combinations[index];
            const score = evaluateEstimator(this.estimatorFactory, params, X, y, this.cv, this.scoring);
            if (score > this.bestScore) {
                this.bestScore = score;
                this.bestParams = { ...params };
            }
        }

        if (this.refit && this.bestParams) {
            this.bestEstimator = this.estimatorFactory(this.bestParams);
            this.bestEstimator.fit(X, y);
        }
    }

    public predict(X: number[][]): number[] {
        if (!this.bestEstimator) {
            throw new Error('search must be fitted before calling predict');
        }
        return this.bestEstimator.predict(X);
    }

    public score(X: number[][], y: number[]): number {
        if (!this.bestEstimator) {
            throw new Error('search must be fitted before calling score');
        }
        if (this.scoring) {
            return this.scoring(this.bestEstimator.predict(X), y);
        }
        if (typeof this.bestEstimator.score === 'function') {
            return this.bestEstimator.score(X, y);
        }
        return accuracyScore(this.bestEstimator.predict(X), y);
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

    const splitter = resolveSplitter(options.cv);

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
