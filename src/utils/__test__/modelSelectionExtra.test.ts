import { LogisticRegression } from '../../linear/logisticRegression';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { Pipeline } from '../../pipeline/pipeline';
import { loadModel } from '../../base/estimator';
import {
    FoldIndices,
    GridSearchCV,
    GroupKFold,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    crossValidate,
    learningCurve,
    validationCurve,
} from '../modelSelection';
import { trainTestSplit } from '../sampling';

function sortedTest(fold: FoldIndices): number[] {
    return fold.testIndices.slice().sort((a, b) => a - b);
}

function assertDisjointAndInRange(fold: FoldIndices, n: number): void {
    const trainSet = new Set(fold.trainIndices);
    for (const i of fold.testIndices) {
        expect(trainSet.has(i)).toBe(false);
    }
    for (const i of [...fold.trainIndices, ...fold.testIndices]) {
        expect(i).toBeGreaterThanOrEqual(0);
        expect(i).toBeLessThan(n);
    }
    expect(new Set(fold.trainIndices).size).toBe(fold.trainIndices.length);
    expect(new Set(fold.testIndices).size).toBe(fold.testIndices.length);
}

// ---------------------------------------------------------------------------
// ShuffleSplit
// ---------------------------------------------------------------------------

test('ShuffleSplit produces nSplits disjoint folds with sklearn sizing', () => {
    const X = Array.from({ length: 10 }, (_, i) => [i]);
    const ss = new ShuffleSplit({ nSplits: 5, testSize: 0.3, randomState: 42 });
    const folds = ss.split(X);

    expect(folds).toHaveLength(5);
    for (const fold of folds) {
        expect(fold.testIndices).toHaveLength(3); // ceil(0.3 * 10)
        expect(fold.trainIndices).toHaveLength(7);
        assertDisjointAndInRange(fold, 10);
    }
    // successive splits differ
    expect(sortedTest(folds[0])).not.toEqual(sortedTest(folds[1]));
});

test('ShuffleSplit is deterministic with randomState and honors trainSize', () => {
    const X = Array.from({ length: 10 }, (_, i) => [i]);
    const a = new ShuffleSplit({ nSplits: 4, testSize: 0.2, trainSize: 0.5, randomState: 7 });
    const b = new ShuffleSplit({ nSplits: 4, testSize: 0.2, trainSize: 0.5, randomState: 7 });

    const foldsA = a.split(X);
    expect(foldsA).toEqual(b.split(X));
    for (const fold of foldsA) {
        expect(fold.testIndices).toHaveLength(2); // ceil(0.2 * 10)
        expect(fold.trainIndices).toHaveLength(5); // floor(0.5 * 10)
    }
});

test('ShuffleSplit defaults to testSize 0.1 and 10 splits', () => {
    const X = Array.from({ length: 10 }, (_, i) => [i]);
    const folds = new ShuffleSplit({ randomState: 1 }).split(X);
    expect(folds).toHaveLength(10);
    for (const fold of folds) {
        expect(fold.testIndices).toHaveLength(1);
        expect(fold.trainIndices).toHaveLength(9);
    }
});

// ---------------------------------------------------------------------------
// StratifiedShuffleSplit
// ---------------------------------------------------------------------------

test('StratifiedShuffleSplit preserves class proportions in every split', () => {
    const n = 20;
    const X = Array.from({ length: n }, (_, i) => [i]);
    const y = Array.from({ length: n }, (_, i) => (i < 12 ? 0 : 1)); // 12 vs 8

    const sss = new StratifiedShuffleSplit({ nSplits: 6, testSize: 0.25, randomState: 3 });
    const folds = sss.split(X, y);

    expect(folds).toHaveLength(6);
    for (const fold of folds) {
        expect(fold.testIndices).toHaveLength(5);
        expect(fold.trainIndices).toHaveLength(15);
        assertDisjointAndInRange(fold, n);
        const testLabels = fold.testIndices.map(i => y[i]);
        expect(testLabels.filter(l => l === 0)).toHaveLength(3); // 12/20 * 5
        expect(testLabels.filter(l => l === 1)).toHaveLength(2); // 8/20 * 5
        const trainLabels = fold.trainIndices.map(i => y[i]);
        expect(trainLabels.filter(l => l === 0)).toHaveLength(9); // 12/20 * 15
        expect(trainLabels.filter(l => l === 1)).toHaveLength(6);
    }
});

test('StratifiedShuffleSplit is deterministic with randomState and validates y', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i]);
    const y = X.map((_, i) => i % 2);
    const a = new StratifiedShuffleSplit({ nSplits: 3, testSize: 4, randomState: 11 });
    const b = new StratifiedShuffleSplit({ nSplits: 3, testSize: 4, randomState: 11 });
    expect(a.split(X, y)).toEqual(b.split(X, y));

    expect(() => new StratifiedShuffleSplit().split(X)).toThrow('requires y labels');
    expect(() => new StratifiedShuffleSplit({ testSize: 0.5 }).split([[0], [1], [2]], [0, 0, 1]))
        .toThrow('too few');
});

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

test('GroupKFold reproduces the sklearn docs example', () => {
    // sklearn: X 4 samples, groups [0, 0, 2, 2], n_splits=2
    //   TRAIN: [0 1] TEST: [2 3] / TRAIN: [2 3] TEST: [0 1]
    const X = [[1, 2], [3, 4], [5, 6], [7, 8]];
    const y = [1, 2, 3, 4];
    const folds = new GroupKFold({ nSplits: 2 }).split(X, y, [0, 0, 2, 2]);

    expect(folds).toHaveLength(2);
    expect(folds[0].testIndices).toEqual([2, 3]);
    expect(folds[0].trainIndices).toEqual([0, 1]);
    expect(folds[1].testIndices).toEqual([0, 1]);
    expect(folds[1].trainIndices).toEqual([2, 3]);
});

test('GroupKFold keeps groups disjoint across train/test and covers all samples once', () => {
    const groups = ['a', 'b', 'a', 'c', 'd', 'b', 'e', 'c', 'd', 'e', 'a', 'f'];
    const X = groups.map((_, i) => [i]);
    const folds = new GroupKFold({ nSplits: 3 }).split(X, undefined, groups);

    expect(folds).toHaveLength(3);
    const seenInTest: number[] = [];
    for (const fold of folds) {
        assertDisjointAndInRange(fold, X.length);
        const testGroups = new Set(fold.testIndices.map(i => groups[i]));
        for (const i of fold.trainIndices) {
            expect(testGroups.has(groups[i])).toBe(false);
        }
        seenInTest.push(...fold.testIndices);
    }
    expect(seenInTest.sort((a, b) => a - b)).toEqual(X.map((_, i) => i));
});

test('GroupKFold validates its inputs', () => {
    const X = [[0], [1], [2], [3]];
    expect(() => new GroupKFold({ nSplits: 2 }).split(X)).toThrow('requires a groups array');
    expect(() => new GroupKFold({ nSplits: 2 }).split(X, undefined, [0, 0])).toThrow('same length');
    expect(() => new GroupKFold({ nSplits: 3 }).split(X, undefined, [0, 0, 1, 1]))
        .toThrow('greater than the number of groups');
});

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

test('TimeSeriesSplit(nSplits=3) on 6 samples matches sklearn exactly', () => {
    const X = Array.from({ length: 6 }, (_, i) => [i]);
    const folds = new TimeSeriesSplit({ nSplits: 3 }).split(X);

    expect(folds).toEqual([
        { trainIndices: [0, 1, 2], testIndices: [3] },
        { trainIndices: [0, 1, 2, 3], testIndices: [4] },
        { trainIndices: [0, 1, 2, 3, 4], testIndices: [5] },
    ]);
});

test('TimeSeriesSplit honors testSize, gap and maxTrainSize like sklearn', () => {
    const X = Array.from({ length: 10 }, (_, i) => [i]);

    expect(new TimeSeriesSplit({ nSplits: 2, testSize: 2, gap: 2 }).split(X)).toEqual([
        { trainIndices: [0, 1, 2, 3], testIndices: [6, 7] },
        { trainIndices: [0, 1, 2, 3, 4, 5], testIndices: [8, 9] },
    ]);

    expect(new TimeSeriesSplit({ nSplits: 2, testSize: 2, gap: 2, maxTrainSize: 3 }).split(X)).toEqual([
        { trainIndices: [1, 2, 3], testIndices: [6, 7] },
        { trainIndices: [3, 4, 5], testIndices: [8, 9] },
    ]);
});

test('TimeSeriesSplit rejects configurations that leave no training data', () => {
    const X = Array.from({ length: 6 }, (_, i) => [i]);
    expect(() => new TimeSeriesSplit({ nSplits: 3, testSize: 2 }).split(X)).toThrow('Too many splits');
    expect(() => new TimeSeriesSplit({ nSplits: 6 }).split(X)).toThrow('greater than the number of samples');
});

// ---------------------------------------------------------------------------
// RepeatedKFold / RepeatedStratifiedKFold
// ---------------------------------------------------------------------------

test('RepeatedKFold yields nSplits * nRepeats folds with distinct shuffles per repeat', () => {
    const X = Array.from({ length: 8 }, (_, i) => [i]);
    const rkf = new RepeatedKFold({ nSplits: 2, nRepeats: 3, randomState: 5 });
    const folds = rkf.split(X);

    expect(folds).toHaveLength(6);
    // each repeat is a full partition of the samples
    for (let r = 0; r < 3; r++) {
        const repeat = folds.slice(r * 2, r * 2 + 2);
        const covered = repeat.flatMap(f => f.testIndices).sort((a, b) => a - b);
        expect(covered).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        repeat.forEach(fold => assertDisjointAndInRange(fold, 8));
    }
    // repeats use different shuffles (signatures of the first fold's test set differ somewhere)
    const signatures = [0, 1, 2].map(r => JSON.stringify(sortedTest(folds[r * 2])));
    expect(new Set(signatures).size).toBeGreaterThan(1);
    // deterministic given the seed
    expect(new RepeatedKFold({ nSplits: 2, nRepeats: 3, randomState: 5 }).split(X)).toEqual(folds);
});

test('RepeatedStratifiedKFold stratifies every fold of every repeat', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i]);
    const y = X.map((_, i) => (i < 6 ? 0 : 1));
    const folds = new RepeatedStratifiedKFold({ nSplits: 3, nRepeats: 2, randomState: 8 }).split(X, y);

    expect(folds).toHaveLength(6);
    for (const fold of folds) {
        const testLabels = fold.testIndices.map(i => y[i]).sort((a, b) => a - b);
        expect(testLabels).toEqual([0, 0, 1, 1]);
    }
});

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

test('LeaveOneOut produces one single-sample test fold per sample', () => {
    const X = [[0], [1], [2], [3]];
    const folds = new LeaveOneOut().split(X);

    expect(folds).toEqual([
        { trainIndices: [1, 2, 3], testIndices: [0] },
        { trainIndices: [0, 2, 3], testIndices: [1] },
        { trainIndices: [0, 1, 3], testIndices: [2] },
        { trainIndices: [0, 1, 2], testIndices: [3] },
    ]);
});

// ---------------------------------------------------------------------------
// splitters serialize inside search objects
// ---------------------------------------------------------------------------

test('new splitters serialize and revive inside a search estimator', () => {
    const X = [[0], [10], [1], [11], [2], [12], [3], [13]];
    const y = [0, 1, 0, 1, 0, 1, 0, 1];
    const search = new GridSearchCV({
        estimator: new LogisticRegression({ maxIter: 100 }),
        paramGrid: { learningRate: [0.1, 0.5] },
        cv: new ShuffleSplit({ nSplits: 3, testSize: 0.25, randomState: 1 }),
        scoring: 'accuracyScore',
    });
    search.fit(X, y);

    const revived = loadModel(JSON.stringify(search)) as GridSearchCV;
    expect((revived.getParams().cv as object).constructor).toBe(ShuffleSplit);
    expect(revived.predict([[0.5], [11.5]])).toEqual(search.predict([[0.5], [11.5]]));

    // remaining splitters round-trip through the tagged-JSON codec as cv params
    const splitters = [
        new StratifiedShuffleSplit({ nSplits: 2, testSize: 0.25, randomState: 2 }),
        new GroupKFold({ nSplits: 2 }),
        new TimeSeriesSplit({ nSplits: 2, gap: 1 }),
        new RepeatedKFold({ nSplits: 2, nRepeats: 2, randomState: 3 }),
        new RepeatedStratifiedKFold({ nSplits: 2, nRepeats: 2, randomState: 3 }),
        new LeaveOneOut(),
    ];
    for (const splitter of splitters) {
        const holder = new GridSearchCV({
            estimator: new LogisticRegression(),
            paramGrid: { learningRate: [0.1] },
            cv: splitter,
        });
        const back = loadModel(JSON.stringify(holder)) as GridSearchCV;
        expect((back.getParams().cv as object).constructor).toBe(splitter.constructor);
    }
});

// ---------------------------------------------------------------------------
// crossValidate
// ---------------------------------------------------------------------------

const CV_X = [[0], [10], [1], [11], [2], [12], [3], [13]];
const CV_Y = [0, 1, 0, 1, 0, 1, 0, 1];

test('crossValidate returns multi-metric test scores keyed by name', () => {
    const result = crossValidate(new LogisticRegression({ maxIter: 200 }), CV_X, CV_Y, {
        cv: 4,
        scoring: ['accuracyScore', 'f1Score'],
    });

    expect(Object.keys(result.testScore).sort()).toEqual(['accuracyScore', 'f1Score']);
    expect(result.testScore.accuracyScore).toHaveLength(4);
    expect(result.testScore.f1Score).toHaveLength(4);
    result.testScore.accuracyScore.forEach(score => expect(score).toBe(1));
    expect(result.fitTimeMs).toHaveLength(4);
    result.fitTimeMs.forEach(t => expect(t).toBeGreaterThanOrEqual(0));
    expect(result.trainScore).toBeUndefined();
});

test('crossValidate supports a scoring record mixing names and functions', () => {
    const result = crossValidate(new LogisticRegression({ maxIter: 200 }), CV_X, CV_Y, {
        cv: 2,
        scoring: {
            acc: 'accuracyScore',
            errors: (actual, expected) => -actual.filter((v, i) => v !== expected[i]).length,
        },
        returnTrainScore: true,
    });

    expect(Object.keys(result.testScore).sort()).toEqual(['acc', 'errors']);
    expect(result.trainScore).toBeDefined();
    expect(Object.keys(result.trainScore!).sort()).toEqual(['acc', 'errors']);
    expect(result.trainScore!.acc).toHaveLength(2);
    result.testScore.errors.forEach(score => expect(score).toBeLessThanOrEqual(0));
});

test('crossValidate defaults to the estimator score under the "score" key', () => {
    const result = crossValidate(new LogisticRegression({ maxIter: 200 }), CV_X, CV_Y, { cv: 2 });
    expect(Object.keys(result.testScore)).toEqual(['score']);
    expect(result.testScore.score).toHaveLength(2);
});

test('crossValidate validates its inputs', () => {
    expect(() => crossValidate({ fit() {}, predict: () => [] } as any, CV_X, CV_Y))
        .toThrow('contract estimator');
    expect(() => crossValidate(new LogisticRegression(), CV_X, CV_Y.slice(1)))
        .toThrow('same length');
    expect(() => crossValidate(new LogisticRegression(), CV_X, CV_Y, { scoring: 'nope' }))
        .toThrow('Unknown scoring');
});

// ---------------------------------------------------------------------------
// learningCurve
// ---------------------------------------------------------------------------

test('learningCurve returns monotone sizes and sizes x folds score matrices', () => {
    // two interleaved separable blobs, 20 samples
    const X = Array.from({ length: 20 }, (_, i) => [i % 2 === 0 ? i : 100 + i]);
    const y = X.map((_, i) => i % 2);

    const result = learningCurve(new DecisionTreeClassifier({ max_depth: 3 }), X, y, {
        cv: 4,
        scoring: 'accuracyScore',
    });

    // classifiers default to StratifiedKFold (sklearn semantics): fold 0's
    // training part holds 14 samples -> floor([1.4, 4.55, 7.7, 10.85, 14])
    expect(result.trainSizesAbs).toEqual([1, 4, 7, 10, 14]);
    for (let s = 1; s < result.trainSizesAbs.length; s++) {
        expect(result.trainSizesAbs[s]).toBeGreaterThan(result.trainSizesAbs[s - 1]);
    }
    expect(result.trainScores).toHaveLength(5);
    expect(result.testScores).toHaveLength(5);
    for (const row of [...result.trainScores, ...result.testScores]) {
        expect(row).toHaveLength(4);
        row.forEach(score => {
            expect(score).toBeGreaterThanOrEqual(0);
            expect(score).toBeLessThanOrEqual(1);
        });
    }
    // the full training fold is separable: perfect train accuracy at max size
    result.trainScores[4].forEach(score => expect(score).toBe(1));
});

test('learningCurve shuffle is seeded and deterministic', () => {
    const X = Array.from({ length: 16 }, (_, i) => [i % 2 === 0 ? i : 100 + i]);
    const y = X.map((_, i) => i % 2);
    const opts = {
        trainSizes: [0.5, 1.0],
        cv: 4,
        scoring: 'accuracyScore' as const,
        shuffle: true,
        randomState: 21,
    };

    const a = learningCurve(new DecisionTreeClassifier({ max_depth: 3 }), X, y, opts);
    const b = learningCurve(new DecisionTreeClassifier({ max_depth: 3 }), X, y, opts);
    expect(a).toEqual(b);
    expect(a.trainSizesAbs).toEqual([6, 12]);
});

// ---------------------------------------------------------------------------
// validationCurve
// ---------------------------------------------------------------------------

test('validationCurve sweeps a hyperparameter and the scores respond', () => {
    // XOR pattern repeated 4x: depth 1 cannot fit it, depth 4 can
    const pattern: Array<[number[], number]> = [
        [[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 0],
    ];
    const X: number[][] = [];
    const y: number[] = [];
    for (let r = 0; r < 4; r++) {
        for (const [xi, yi] of pattern) {
            X.push(xi);
            y.push(yi);
        }
    }

    const result = validationCurve(new DecisionTreeClassifier(), X, y, {
        paramName: 'max_depth',
        paramRange: [1, 4],
        cv: 4,
        scoring: 'accuracyScore',
    });

    expect(result.trainScores).toHaveLength(2);
    expect(result.testScores).toHaveLength(2);
    result.trainScores.forEach(row => expect(row).toHaveLength(4));

    const meanShallow = result.trainScores[0].reduce((a, b) => a + b, 0) / 4;
    const meanDeep = result.trainScores[1].reduce((a, b) => a + b, 0) / 4;
    expect(meanDeep).toBe(1); // depth 4 memorizes XOR
    expect(meanShallow).toBeLessThan(meanDeep);
});

test('validationCurve supports pipeline-style step__param addressing', () => {
    const X = Array.from({ length: 16 }, (_, i) => [i % 2 === 0 ? i : 100 + i, 1]);
    const y = X.map((_, i) => i % 2);
    const pipe = new Pipeline({
        steps: [
            ['clf', new DecisionTreeClassifier()],
        ],
    });

    const result = validationCurve(pipe as any, X, y, {
        paramName: 'clf__max_depth',
        paramRange: [1, 3],
        cv: 4,
        scoring: 'accuracyScore',
    });

    expect(result.trainScores).toHaveLength(2);
    expect(result.testScores).toHaveLength(2);
    // the data is separable on the first feature, so both depths fit the training folds
    result.trainScores[1].forEach(score => expect(score).toBe(1));
});

test('validationCurve validates its options', () => {
    expect(() => validationCurve(new DecisionTreeClassifier(), CV_X, CV_Y, {
        paramName: '',
        paramRange: [1],
    })).toThrow('paramName');
    expect(() => validationCurve(new DecisionTreeClassifier(), CV_X, CV_Y, {
        paramName: 'max_depth',
        paramRange: [],
    })).toThrow('paramRange');
});

// ---------------------------------------------------------------------------
// trainTestSplit (extended: trainSize + stratify)
// ---------------------------------------------------------------------------

test('trainTestSplit stratify preserves class proportions and is seeded', () => {
    const n = 20;
    const X = Array.from({ length: n }, (_, i) => [i]);
    const y = Array.from({ length: n }, (_, i) => (i % 5 < 3 ? 0 : 1)); // 12 vs 8

    const run1 = trainTestSplit(X, y, { testSize: 0.25, randomState: 42, stratify: y });
    const run2 = trainTestSplit(X, y, { testSize: 0.25, randomState: 42, stratify: y });
    expect(run1).toEqual(run2);

    expect(run1.XTest).toHaveLength(5);
    expect(run1.XTrain).toHaveLength(15);
    expect(run1.yTest!.filter(l => l === 0)).toHaveLength(3);
    expect(run1.yTest!.filter(l => l === 1)).toHaveLength(2);
    expect(run1.yTrain!.filter(l => l === 0)).toHaveLength(9);
    expect(run1.yTrain!.filter(l => l === 1)).toHaveLength(6);

    // rows keep their labels (X index i carries label y[i])
    run1.XTest!.forEach((row, k) => expect(y[row[0]]).toBe(run1.yTest![k]));

    // no sample appears in both sides
    const trainSet = new Set(run1.XTrain.map(row => row[0]));
    run1.XTest.forEach(row => expect(trainSet.has(row[0])).toBe(false));
});

test('trainTestSplit supports trainSize and validates stratify usage', () => {
    const X = Array.from({ length: 20 }, (_, i) => [i]);
    const y = X.map((_, i) => i % 2);

    const result = trainTestSplit(X, y, { testSize: 0.2, trainSize: 0.5, randomState: 1 });
    expect(result.XTest).toHaveLength(4); // ceil(0.2 * 20)
    expect(result.XTrain).toHaveLength(10); // floor(0.5 * 20)

    expect(() => trainTestSplit(X, y, { stratify: y, shuffle: false })).toThrow('requires shuffle');
    expect(() => trainTestSplit(X, y, { stratify: y.slice(1) })).toThrow('same length');
});
