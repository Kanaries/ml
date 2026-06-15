import { KNearestNeighbors } from '../../neighbors/knn';
import { meanSquaredError } from '../../metrics';
import { KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, crossValScore } from '../modelSelection';

test('KFold split without shuffle is deterministic and exhaustive', () => {
    const X = Array.from({ length: 10 }, (_, i) => [i]);
    const kf = new KFold({ nSplits: 5 });
    const folds = kf.split(X);

    expect(folds).toHaveLength(5);
    const testAll = folds.flatMap(fold => fold.testIndices).sort((a, b) => a - b);
    expect(testAll).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    folds.forEach(fold => {
        expect(fold.trainIndices.length).toBe(8);
        expect(fold.testIndices.length).toBe(2);
    });
});

test('KFold split with shuffle honors randomState', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i]);
    const kf1 = new KFold({ nSplits: 3, shuffle: true, randomState: 7 });
    const kf2 = new KFold({ nSplits: 3, shuffle: true, randomState: 7 });

    expect(kf1.split(X)).toEqual(kf2.split(X));
});

test('crossValScore computes per-fold scores with estimator score method', () => {
    const X = [[0], [1], [2], [3], [10], [11], [12], [13]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const scores = crossValScore(
        () => new KNearestNeighbors(1),
        X,
        y,
        { cv: new KFold({ nSplits: 4, shuffle: true, randomState: 42 }) },
    );

    expect(scores).toHaveLength(4);
    scores.forEach(score => {
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
    });
});

test('crossValScore supports custom scoring callback', () => {
    const X = [[0], [1], [2], [3], [4], [5]];
    const y = [0, 1, 2, 3, 4, 5];

    class MeanRegressor {
        private mean = 0;
        fit(_trainX: number[][], trainY: number[]) {
            this.mean = trainY.reduce((acc, v) => acc + v, 0) / trainY.length;
        }
        predict(testX: number[][]): number[] {
            return testX.map(() => this.mean);
        }
    }

    const scores = crossValScore(
        () => new MeanRegressor(),
        X,
        y,
        {
            cv: 3,
            scoring: (actual, expected) => -meanSquaredError(actual, expected),
        },
    );

    expect(scores).toHaveLength(3);
    scores.forEach(score => {
        expect(Number.isFinite(score)).toBe(true);
        expect(score).toBeLessThanOrEqual(0);
    });
});

test('KFold and crossValScore validate inputs', () => {
    expect(() => new KFold({ nSplits: 1 })).toThrow('nSplits must be an integer >= 2');
    expect(() => new KFold({ nSplits: 5 }).split([[1], [2]])).toThrow('nSplits cannot be greater than number of samples');
    expect(() =>
        crossValScore(() => new KNearestNeighbors(1), [[1], [2]], [0], { cv: 2 }),
    ).toThrow('X and y must have the same length');
});

test('StratifiedKFold preserves class balance across folds', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i]);
    const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    const skf = new StratifiedKFold({ nSplits: 3, shuffle: true, randomState: 7 });
    const folds = skf.split(X, y);

    expect(folds).toHaveLength(3);
    for (const fold of folds) {
        const testLabels = fold.testIndices.map(index => y[index]).sort((a, b) => a - b);
        expect(testLabels).toEqual([0, 0, 1, 1]);
    }
});

test('StratifiedKFold validates label distribution', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 0, 0, 1];
    expect(() => new StratifiedKFold({ nSplits: 3 }).split(X, y)).toThrow(
        'Each class must have at least nSplits samples',
    );
});

test('GridSearchCV selects the best parameter combination and refits', () => {
    const X = [[0], [1], [2], [10], [11], [12]];
    const y = [0, 0, 0, 1, 1, 1];

    const search = new GridSearchCV({
        estimatorFactory: params => new KNearestNeighbors(params.nNeighbors),
        paramGrid: {
            nNeighbors: [1, 3, 5],
        },
        cv: new KFold({ nSplits: 3, shuffle: true, randomState: 42 }),
    });

    search.fit(X, y);

    expect(search.bestParams).toEqual({ nNeighbors: 1 });
    expect(search.bestScore).toBeGreaterThanOrEqual(0);
    expect(search.predict([[0.2], [11.8]])).toEqual([0, 1]);
});

test('RandomizedSearchCV is reproducible with randomState', () => {
    const X = [[0], [1], [2], [10], [11], [12]];
    const y = [0, 0, 0, 1, 1, 1];
    const config = {
        estimatorFactory: (params: { nNeighbors: number }) => new KNearestNeighbors(params.nNeighbors),
        paramDistributions: {
            nNeighbors: [1, 3, 5],
        },
        nIter: 2,
        cv: new KFold({ nSplits: 3, shuffle: true, randomState: 42 }),
        randomState: 9,
    };

    const run1 = new RandomizedSearchCV(config);
    const run2 = new RandomizedSearchCV(config);
    run1.fit(X, y);
    run2.fit(X, y);

    expect(run1.bestParams).toEqual(run2.bestParams);
    expect(run1.bestScore).toBeCloseTo(run2.bestScore, 12);
});

test('search estimators validate fit lifecycle', () => {
    const search = new GridSearchCV({
        estimatorFactory: params => new KNearestNeighbors(params.nNeighbors),
        paramGrid: { nNeighbors: [1] },
    });

    expect(() => search.predict([[0]])).toThrow('search must be fitted before calling predict');
});
