/**
 * Extended model-selection utilities (sklearn-matching semantics):
 *  - splitters: ShuffleSplit, StratifiedShuffleSplit, GroupKFold,
 *    TimeSeriesSplit, RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut;
 *  - crossValidate (multi-metric cross-validation with fit timing);
 *  - learningCurve / validationCurve.
 *
 * This module is re-exported from `./modelSelection` (single import surface).
 * All value imports from `./modelSelection` are only referenced inside
 * function bodies so the circular re-export stays initialization-safe.
 */
import { BaseEstimator, registerSerializableClass } from '../base/estimator';
import { ClassifierBase } from '../base/classifier';
import { createRandomGenerator } from './random';
import {
    EstimatorLike,
    FoldIndices,
    KFold,
    SCORING_FUNCS,
    ScoringFunction,
    SearchEstimator,
    SplitterLike,
    StratifiedKFold,
    resolveScoring,
} from './modelSelection';
import { allocateProportionally, trainTestSplit } from './sampling';

// sklearn keeps train_test_split in model_selection; ours lives in sampling.ts
// (it predates this module) and is re-exported here for discoverability.
export { trainTestSplit };
export type { TrainTestSplitOptions, TrainTestSplitResult } from './sampling';

function shuffledIndices(size: number, random: () => number): number[] {
    const indices = Array.from({ length: size }, (_, i) => i);
    for (let i = size - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

function range(start: number, end: number): number[] {
    return Array.from({ length: end - start }, (_, i) => start + i);
}

/** Derive a fresh integer seed for a nested seeded splitter. */
function drawSeed(random: () => number): number {
    return Math.floor(random() * 2147483646) + 1;
}

function validateXy(X: any[], y?: any[]): void {
    if (X.length < 2) {
        throw new Error('X must contain at least 2 samples');
    }
    if (y && y.length !== X.length) {
        throw new Error('X and y must have the same length');
    }
}

/**
 * sklearn `_validate_shuffle_split`: float testSize rounds up, float
 * trainSize rounds down, missing sizes take the complement (or
 * `defaultTestSize` when both are missing).
 */
function resolveShuffleSizes(
    sampleCount: number,
    testSize: number | undefined,
    trainSize: number | undefined,
    defaultTestSize: number,
): { nTest: number; nTrain: number } {
    if (testSize === undefined && trainSize === undefined) {
        testSize = defaultTestSize;
    }
    let nTest: number | undefined;
    if (testSize !== undefined) {
        if (!Number.isFinite(testSize) || testSize <= 0) {
            throw new Error('testSize must be a positive finite number');
        }
        if (testSize >= 1 && !Number.isInteger(testSize)) {
            throw new Error('absolute testSize must be an integer');
        }
        nTest = testSize < 1 ? Math.ceil(sampleCount * testSize) : testSize;
    }
    let nTrain: number | undefined;
    if (trainSize !== undefined) {
        if (!Number.isFinite(trainSize) || trainSize <= 0) {
            throw new Error('trainSize must be a positive finite number');
        }
        if (trainSize >= 1 && !Number.isInteger(trainSize)) {
            throw new Error('absolute trainSize must be an integer');
        }
        nTrain = trainSize < 1 ? Math.floor(sampleCount * trainSize) : trainSize;
    }
    if (nTest === undefined) {
        nTest = sampleCount - nTrain!;
    }
    if (nTrain === undefined) {
        nTrain = sampleCount - nTest;
    }
    if (nTest < 1 || nTrain < 1 || nTest + nTrain > sampleCount) {
        throw new Error(`train/test sizes are invalid: nTrain=${nTrain} + nTest=${nTest} ` +
            `must fit in ${sampleCount} samples with at least 1 sample each`);
    }
    return { nTest, nTrain };
}

function groupByLabel(y: any[]): Map<any, number[]> {
    const labelToIndices = new Map<any, number[]>();
    for (let i = 0; i < y.length; i++) {
        if (!labelToIndices.has(y[i])) {
            labelToIndices.set(y[i], []);
        }
        labelToIndices.get(y[i])!.push(i);
    }
    return labelToIndices;
}

// ---------------------------------------------------------------------------
// ShuffleSplit / StratifiedShuffleSplit
// ---------------------------------------------------------------------------

export interface ShuffleSplitProps {
    nSplits?: number;
    /** Fraction in (0, 1) or absolute count. Defaults to 0.1 when trainSize is unset. */
    testSize?: number;
    /** Fraction in (0, 1) or absolute count. Defaults to the complement of testSize. */
    trainSize?: number;
    randomState?: number;
}

/** Random permutation cross-validator (sklearn `ShuffleSplit`). */
export class ShuffleSplit implements SplitterLike {
    private nSplits: number;
    private testSize?: number;
    private trainSize?: number;
    private randomState?: number;

    constructor(props: ShuffleSplitProps = {}) {
        const { nSplits = 10, testSize, trainSize, randomState } = props;
        if (!Number.isInteger(nSplits) || nSplits < 1) {
            throw new Error('nSplits must be an integer >= 1');
        }
        this.nSplits = nSplits;
        this.testSize = testSize;
        this.trainSize = trainSize;
        this.randomState = randomState;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        validateXy(X, y);
        const { nTest, nTrain } = resolveShuffleSizes(X.length, this.testSize, this.trainSize, 0.1);
        const random = createRandomGenerator(this.randomState);
        const folds: FoldIndices[] = [];
        for (let s = 0; s < this.nSplits; s++) {
            const permutation = shuffledIndices(X.length, random);
            folds.push({
                testIndices: permutation.slice(0, nTest),
                trainIndices: permutation.slice(nTest, nTest + nTrain),
            });
        }
        return folds;
    }
}
registerSerializableClass('utils.ShuffleSplit', ShuffleSplit);

/**
 * Stratified random permutation cross-validator (sklearn
 * `StratifiedShuffleSplit`): each split preserves per-class proportions.
 * Per-class counts use deterministic largest-remainder rounding (sklearn
 * randomizes the rounding of remainders), so proportions match sklearn but
 * exact index draws do not.
 */
export class StratifiedShuffleSplit implements SplitterLike {
    private nSplits: number;
    private testSize?: number;
    private trainSize?: number;
    private randomState?: number;

    constructor(props: ShuffleSplitProps = {}) {
        const { nSplits = 10, testSize, trainSize, randomState } = props;
        if (!Number.isInteger(nSplits) || nSplits < 1) {
            throw new Error('nSplits must be an integer >= 1');
        }
        this.nSplits = nSplits;
        this.testSize = testSize;
        this.trainSize = trainSize;
        this.randomState = randomState;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        if (!y) {
            throw new Error('StratifiedShuffleSplit requires y labels');
        }
        validateXy(X, y);
        const { nTest, nTrain } = resolveShuffleSizes(X.length, this.testSize, this.trainSize, 0.1);

        const labelToIndices = groupByLabel(y);
        const classIndices = Array.from(labelToIndices.values());
        const counts = classIndices.map(indices => indices.length);
        for (const count of counts) {
            if (count < 2) {
                throw new Error('The least populated class in y has only 1 member, which is too few (minimum 2)');
            }
        }
        if (nTrain < counts.length || nTest < counts.length) {
            throw new Error('Both train and test sets must contain at least one sample per class; ' +
                'increase testSize/trainSize or reduce the number of classes');
        }

        const random = createRandomGenerator(this.randomState);
        const folds: FoldIndices[] = [];
        for (let s = 0; s < this.nSplits; s++) {
            // reallocate per split with randomized tie-breaking so no class is
            // systematically over-represented across folds (sklearn behavior)
            const trainAlloc = allocateProportionally(counts, nTrain, random);
            const testAlloc = allocateProportionally(counts.map((c, i) => c - trainAlloc[i]), nTest, random);
            const trainIndices: number[] = [];
            const testIndices: number[] = [];
            for (let c = 0; c < classIndices.length; c++) {
                const shuffled = shuffledIndices(classIndices[c].length, random).map(i => classIndices[c][i]);
                trainIndices.push(...shuffled.slice(0, trainAlloc[c]));
                testIndices.push(...shuffled.slice(trainAlloc[c], trainAlloc[c] + testAlloc[c]));
            }
            const trainOrder = shuffledIndices(trainIndices.length, random);
            const testOrder = shuffledIndices(testIndices.length, random);
            folds.push({
                trainIndices: trainOrder.map(i => trainIndices[i]),
                testIndices: testOrder.map(i => testIndices[i]),
            });
        }
        return folds;
    }
}
registerSerializableClass('utils.StratifiedShuffleSplit', StratifiedShuffleSplit);

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

export interface GroupKFoldProps {
    nSplits?: number;
}

/**
 * K-fold with non-overlapping groups (sklearn `GroupKFold`): the same group
 * never appears in both train and test, and fold sizes are balanced by
 * greedily assigning the largest groups to the least-filled fold.
 */
export class GroupKFold implements SplitterLike {
    private nSplits: number;

    constructor(props: GroupKFoldProps = {}) {
        const { nSplits = 5 } = props;
        if (!Number.isInteger(nSplits) || nSplits < 2) {
            throw new Error('nSplits must be an integer >= 2');
        }
        this.nSplits = nSplits;
    }

    public split(X: any[], y?: any[], groups?: any[]): FoldIndices[] {
        if (!groups) {
            throw new Error('GroupKFold requires a groups array');
        }
        if (groups.length !== X.length) {
            throw new Error('X and groups must have the same length');
        }
        validateXy(X, y);

        // unique groups in ascending order (mirrors np.unique's encoding)
        const uniqueGroups = Array.from(new Set(groups)).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
        if (uniqueGroups.length < this.nSplits) {
            throw new Error(`Cannot have nSplits=${this.nSplits} greater than the number of groups (${uniqueGroups.length})`);
        }
        const groupToId = new Map<any, number>(uniqueGroups.map((g, i) => [g, i]));
        const groupCounts = new Array<number>(uniqueGroups.length).fill(0);
        for (const g of groups) {
            groupCounts[groupToId.get(g)!]++;
        }

        // sklearn: argsort(counts)[::-1] — count descending, id descending on ties
        const order = groupCounts
            .map((count, id) => ({ count, id }))
            .sort((a, b) => b.count - a.count || b.id - a.id);

        const foldWeights = new Array<number>(this.nSplits).fill(0);
        const groupToFold = new Array<number>(uniqueGroups.length).fill(0);
        for (const { count, id } of order) {
            let lightest = 0;
            for (let f = 1; f < this.nSplits; f++) {
                if (foldWeights[f] < foldWeights[lightest]) {
                    lightest = f;
                }
            }
            foldWeights[lightest] += count;
            groupToFold[id] = lightest;
        }

        const foldTestIndices: number[][] = Array.from({ length: this.nSplits }, () => []);
        for (let i = 0; i < X.length; i++) {
            foldTestIndices[groupToFold[groupToId.get(groups[i])!]].push(i);
        }
        return foldTestIndices.map(testIndices => {
            const testSet = new Set(testIndices);
            return {
                testIndices,
                trainIndices: range(0, X.length).filter(i => !testSet.has(i)),
            };
        });
    }
}
registerSerializableClass('utils.GroupKFold', GroupKFold);

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

export interface TimeSeriesSplitProps {
    nSplits?: number;
    maxTrainSize?: number;
    testSize?: number;
    gap?: number;
}

/**
 * Forward-chaining time-series cross-validator (sklearn `TimeSeriesSplit`):
 * successive training sets are supersets of earlier ones and always precede
 * the test set. No shuffling.
 */
export class TimeSeriesSplit implements SplitterLike {
    private nSplits: number;
    private maxTrainSize?: number;
    private testSize?: number;
    private gap: number;

    constructor(props: TimeSeriesSplitProps = {}) {
        const { nSplits = 5, maxTrainSize, testSize, gap = 0 } = props;
        if (!Number.isInteger(nSplits) || nSplits < 2) {
            throw new Error('nSplits must be an integer >= 2');
        }
        if (maxTrainSize !== undefined && (!Number.isInteger(maxTrainSize) || maxTrainSize < 1)) {
            throw new Error('maxTrainSize must be an integer >= 1');
        }
        if (testSize !== undefined && (!Number.isInteger(testSize) || testSize < 1)) {
            throw new Error('testSize must be an integer >= 1');
        }
        if (!Number.isInteger(gap) || gap < 0) {
            throw new Error('gap must be an integer >= 0');
        }
        this.nSplits = nSplits;
        this.maxTrainSize = maxTrainSize;
        this.testSize = testSize;
        this.gap = gap;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        validateXy(X, y);
        const n = X.length;
        const nFolds = this.nSplits + 1;
        if (nFolds > n) {
            throw new Error(`Cannot have number of folds=${nFolds} greater than the number of samples=${n}`);
        }
        const testSize = this.testSize ?? Math.floor(n / nFolds);
        if (n - this.gap - testSize * this.nSplits <= 0) {
            throw new Error(`Too many splits=${this.nSplits} for number of samples=${n} ` +
                `with testSize=${testSize} and gap=${this.gap}`);
        }

        const folds: FoldIndices[] = [];
        for (let testStart = n - this.nSplits * testSize; testStart < n; testStart += testSize) {
            const trainEnd = testStart - this.gap;
            const trainStart = this.maxTrainSize !== undefined && this.maxTrainSize < trainEnd
                ? trainEnd - this.maxTrainSize
                : 0;
            folds.push({
                trainIndices: range(trainStart, trainEnd),
                testIndices: range(testStart, testStart + testSize),
            });
        }
        return folds;
    }
}
registerSerializableClass('utils.TimeSeriesSplit', TimeSeriesSplit);

// ---------------------------------------------------------------------------
// RepeatedKFold / RepeatedStratifiedKFold
// ---------------------------------------------------------------------------

export interface RepeatedKFoldProps {
    nSplits?: number;
    nRepeats?: number;
    randomState?: number;
}

function repeatedSplit(
    makeSplitter: (seed: number) => SplitterLike,
    nRepeats: number,
    randomState: number | undefined,
    X: any[],
    y?: any[],
): FoldIndices[] {
    const random = createRandomGenerator(randomState);
    const folds: FoldIndices[] = [];
    for (let r = 0; r < nRepeats; r++) {
        folds.push(...makeSplitter(drawSeed(random)).split(X, y));
    }
    return folds;
}

/** Repeats shuffled KFold `nRepeats` times with fresh randomization each repeat. */
export class RepeatedKFold implements SplitterLike {
    private nSplits: number;
    private nRepeats: number;
    private randomState?: number;

    constructor(props: RepeatedKFoldProps = {}) {
        const { nSplits = 5, nRepeats = 10, randomState } = props;
        if (!Number.isInteger(nSplits) || nSplits < 2) {
            throw new Error('nSplits must be an integer >= 2');
        }
        if (!Number.isInteger(nRepeats) || nRepeats < 1) {
            throw new Error('nRepeats must be an integer >= 1');
        }
        this.nSplits = nSplits;
        this.nRepeats = nRepeats;
        this.randomState = randomState;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        return repeatedSplit(
            seed => new KFold({ nSplits: this.nSplits, shuffle: true, randomState: seed }),
            this.nRepeats, this.randomState, X, y,
        );
    }
}
registerSerializableClass('utils.RepeatedKFold', RepeatedKFold);

/** Repeats shuffled StratifiedKFold `nRepeats` times with fresh randomization each repeat. */
export class RepeatedStratifiedKFold implements SplitterLike {
    private nSplits: number;
    private nRepeats: number;
    private randomState?: number;

    constructor(props: RepeatedKFoldProps = {}) {
        const { nSplits = 5, nRepeats = 10, randomState } = props;
        if (!Number.isInteger(nSplits) || nSplits < 2) {
            throw new Error('nSplits must be an integer >= 2');
        }
        if (!Number.isInteger(nRepeats) || nRepeats < 1) {
            throw new Error('nRepeats must be an integer >= 1');
        }
        this.nSplits = nSplits;
        this.nRepeats = nRepeats;
        this.randomState = randomState;
    }

    public split(X: any[], y?: any[]): FoldIndices[] {
        return repeatedSplit(
            seed => new StratifiedKFold({ nSplits: this.nSplits, shuffle: true, randomState: seed }),
            this.nRepeats, this.randomState, X, y,
        );
    }
}
registerSerializableClass('utils.RepeatedStratifiedKFold', RepeatedStratifiedKFold);

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

/** Each sample is used once as a single-element test set (sklearn `LeaveOneOut`). */
export class LeaveOneOut implements SplitterLike {
    public split(X: any[], y?: any[]): FoldIndices[] {
        validateXy(X, y);
        return X.map((_, i) => ({
            trainIndices: range(0, X.length).filter(j => j !== i),
            testIndices: [i],
        }));
    }
}
registerSerializableClass('utils.LeaveOneOut', LeaveOneOut);

// ---------------------------------------------------------------------------
// crossValidate
// ---------------------------------------------------------------------------

/**
 * Multi-metric scoring spec: a built-in name, a scoring function, a list of
 * built-in names, or a record mapping result keys to names/functions.
 */
export type MultiScoring =
    | string
    | ScoringFunction
    | string[]
    | Record<string, string | ScoringFunction>;

export interface CrossValidateOptions {
    cv?: number | SplitterLike;
    scoring?: MultiScoring;
    returnTrainScore?: boolean;
    /** Passed through to group-aware splitters (e.g. GroupKFold). */
    groups?: any[];
}

export interface CrossValidateResult {
    /** Per-metric test scores, one entry per fold. */
    testScore: Record<string, number[]>;
    /** Present only when `returnTrainScore` is true. */
    trainScore?: Record<string, number[]>;
    /** Wall-clock fit time per fold in milliseconds. */
    fitTimeMs: number[];
}

/**
 * Classifier detection for default-CV stratification, sklearn-style. Covers
 * direct classifiers and pipelines whose final step is a classifier (pipelines
 * expose their steps through getParams, so no import cycle is needed).
 */
function isClassifierLike(estimator: BaseEstimator): boolean {
    if (estimator instanceof ClassifierBase) return true;
    const steps = (estimator.getParams() as { steps?: unknown }).steps;
    if (Array.isArray(steps) && steps.length > 0) {
        const last = steps[steps.length - 1];
        return Array.isArray(last) && last[1] instanceof ClassifierBase;
    }
    return false;
}

/**
 * Explicit splitters pass through; integer/undefined cv becomes
 * StratifiedKFold for classifiers and KFold otherwise (sklearn semantics).
 */
function resolveCv(cv: number | SplitterLike | undefined, estimator?: BaseEstimator): SplitterLike {
    if (cv !== undefined && typeof cv !== 'number') return cv;
    const nSplits = typeof cv === 'number' ? cv : 5;
    return estimator !== undefined && isClassifierLike(estimator)
        ? new StratifiedKFold({ nSplits })
        : new KFold({ nSplits });
}

/**
 * Resolve a MultiScoring spec into named scorers. An `undefined` scorer value
 * means "use the estimator's own score() (accuracy fallback)".
 */
function resolveMultiScoring(scoring?: MultiScoring): Record<string, ScoringFunction | undefined> {
    if (scoring === undefined) {
        return { score: undefined };
    }
    if (typeof scoring === 'function') {
        return { score: scoring };
    }
    if (typeof scoring === 'string') {
        return { [scoring]: resolveScoring(scoring) };
    }
    if (Array.isArray(scoring)) {
        if (scoring.length === 0) {
            throw new Error('scoring array must not be empty');
        }
        const out: Record<string, ScoringFunction | undefined> = {};
        for (const name of scoring) {
            out[name] = resolveScoring(name);
        }
        return out;
    }
    const keys = Object.keys(scoring);
    if (keys.length === 0) {
        throw new Error('scoring record must not be empty');
    }
    const out: Record<string, ScoringFunction | undefined> = {};
    for (const key of keys) {
        out[key] = resolveScoring(scoring[key]);
    }
    return out;
}

function scoreWith(estimator: EstimatorLike, X: number[][], y: number[], scoring?: ScoringFunction): number {
    if (scoring) {
        return scoring(estimator.predict(X), y);
    }
    if (typeof estimator.score === 'function') {
        return estimator.score(X, y);
    }
    return SCORING_FUNCS.accuracyScore(estimator.predict(X), y);
}

function now(): number {
    return typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now();
}

/**
 * Cross-validate a contract estimator (cloned per fold), evaluating one or
 * more metrics on each fold and timing each fit.
 */
export function crossValidate(
    estimator: SearchEstimator,
    X: number[][],
    y: number[],
    options: CrossValidateOptions = {},
): CrossValidateResult {
    if (!(estimator instanceof BaseEstimator)) {
        throw new Error('crossValidate requires a contract estimator (BaseEstimator with clone/setParams)');
    }
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }

    const scorers = resolveMultiScoring(options.scoring);
    const folds = resolveCv(options.cv, estimator).split(X, y, options.groups);
    const names = Object.keys(scorers);
    const returnTrainScore = options.returnTrainScore ?? false;

    const testScore: Record<string, number[]> = Object.fromEntries(names.map(name => [name, []]));
    const trainScore: Record<string, number[]> = Object.fromEntries(names.map(name => [name, []]));
    const fitTimeMs: number[] = [];

    for (const fold of folds) {
        const trainX = fold.trainIndices.map(i => X[i]);
        const trainY = fold.trainIndices.map(i => y[i]);
        const testX = fold.testIndices.map(i => X[i]);
        const testY = fold.testIndices.map(i => y[i]);

        const cloned = estimator.clone();
        const start = now();
        cloned.fit(trainX, trainY);
        fitTimeMs.push(now() - start);

        for (const name of names) {
            testScore[name].push(scoreWith(cloned, testX, testY, scorers[name]));
            if (returnTrainScore) {
                trainScore[name].push(scoreWith(cloned, trainX, trainY, scorers[name]));
            }
        }
    }

    const result: CrossValidateResult = { testScore, fitTimeMs };
    if (returnTrainScore) {
        result.trainScore = trainScore;
    }
    return result;
}

// ---------------------------------------------------------------------------
// learningCurve
// ---------------------------------------------------------------------------

export interface LearningCurveOptions {
    /** Fractions in (0, 1] of the max training size, or absolute counts (> 1). */
    trainSizes?: number[];
    cv?: number | SplitterLike;
    scoring?: string | ScoringFunction;
    /** Shuffle each CV training fold before taking incremental subsets. */
    shuffle?: boolean;
    randomState?: number;
}

export interface LearningCurveResult {
    /** Ascending unique absolute training sizes (rows of the score matrices). */
    trainSizesAbs: number[];
    /** trainScores[sizeIndex][foldIndex] */
    trainScores: number[][];
    /** testScores[sizeIndex][foldIndex] */
    testScores: number[][];
}

/**
 * sklearn `learning_curve`: for each CV fold, fit the estimator on growing
 * prefixes of the SAME (optionally shuffled) training fold and score both the
 * training subset and the fold's test set.
 */
export function learningCurve(
    estimator: SearchEstimator,
    X: number[][],
    y: number[],
    options: LearningCurveOptions = {},
): LearningCurveResult {
    if (!(estimator instanceof BaseEstimator)) {
        throw new Error('learningCurve requires a contract estimator (BaseEstimator with clone/setParams)');
    }
    const { trainSizes = [0.1, 0.325, 0.55, 0.775, 1.0], shuffle = false, randomState } = options;
    if (!Array.isArray(trainSizes) || trainSizes.length === 0) {
        throw new Error('trainSizes must be a non-empty array');
    }

    const folds = resolveCv(options.cv, estimator).split(X, y);
    const scoring = resolveScoring(options.scoring);
    const nMaxTraining = folds[0].trainIndices.length;

    // sklearn's _translate_train_sizes: fractions -> floor, clip to [1, max], unique ascending
    const absolute = trainSizes.map(size => {
        if (!Number.isFinite(size) || size <= 0) {
            throw new Error(`trainSizes entries must be positive, got ${size}`);
        }
        const abs = size <= 1 ? Math.floor(size * nMaxTraining) : Math.floor(size);
        return Math.max(1, Math.min(abs, nMaxTraining));
    });
    const trainSizesAbs = Array.from(new Set(absolute)).sort((a, b) => a - b);

    const random = createRandomGenerator(randomState);
    const trainScores: number[][] = trainSizesAbs.map(() => []);
    const testScores: number[][] = trainSizesAbs.map(() => []);

    for (const fold of folds) {
        const trainIdx = shuffle
            ? shuffledIndices(fold.trainIndices.length, random).map(i => fold.trainIndices[i])
            : fold.trainIndices;
        const testX = fold.testIndices.map(i => X[i]);
        const testY = fold.testIndices.map(i => y[i]);

        for (let s = 0; s < trainSizesAbs.length; s++) {
            const subset = trainIdx.slice(0, trainSizesAbs[s]);
            const subsetX = subset.map(i => X[i]);
            const subsetY = subset.map(i => y[i]);
            const cloned = estimator.clone();
            cloned.fit(subsetX, subsetY);
            trainScores[s].push(scoreWith(cloned, subsetX, subsetY, scoring));
            testScores[s].push(scoreWith(cloned, testX, testY, scoring));
        }
    }

    return { trainSizesAbs, trainScores, testScores };
}

// ---------------------------------------------------------------------------
// validationCurve
// ---------------------------------------------------------------------------

export interface ValidationCurveOptions {
    /** Parameter to sweep; supports pipeline-style `step__param` addressing. */
    paramName: string;
    paramRange: any[];
    cv?: number | SplitterLike;
    scoring?: string | ScoringFunction;
}

export interface ValidationCurveResult {
    /** trainScores[paramIndex][foldIndex] */
    trainScores: number[][];
    /** testScores[paramIndex][foldIndex] */
    testScores: number[][];
}

/**
 * sklearn `validation_curve`: train/test scores across a sweep of one
 * hyperparameter, applied per fold via `clone().setParams(...)`.
 */
export function validationCurve(
    estimator: SearchEstimator,
    X: number[][],
    y: number[],
    options: ValidationCurveOptions,
): ValidationCurveResult {
    if (!(estimator instanceof BaseEstimator)) {
        throw new Error('validationCurve requires a contract estimator (BaseEstimator with clone/setParams)');
    }
    const { paramName, paramRange } = options;
    if (typeof paramName !== 'string' || paramName.length === 0) {
        throw new Error('paramName must be a non-empty string');
    }
    if (!Array.isArray(paramRange) || paramRange.length === 0) {
        throw new Error('paramRange must be a non-empty array');
    }

    const folds = resolveCv(options.cv, estimator).split(X, y);
    const scoring = resolveScoring(options.scoring);
    const trainScores: number[][] = paramRange.map(() => []);
    const testScores: number[][] = paramRange.map(() => []);

    for (let p = 0; p < paramRange.length; p++) {
        for (const fold of folds) {
            const trainX = fold.trainIndices.map(i => X[i]);
            const trainY = fold.trainIndices.map(i => y[i]);
            const testX = fold.testIndices.map(i => X[i]);
            const testY = fold.testIndices.map(i => y[i]);

            const cloned = estimator.clone().setParams({ [paramName]: paramRange[p] });
            cloned.fit(trainX, trainY);
            trainScores[p].push(scoreWith(cloned, trainX, trainY, scoring));
            testScores[p].push(scoreWith(cloned, testX, testY, scoring));
        }
    }

    return { trainScores, testScores };
}
