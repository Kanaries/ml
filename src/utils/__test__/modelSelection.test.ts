import { KNearstNeighbors } from '../../neighbors/knn';
import { meanSquaredError } from '../../metrics';
import { KFold, crossValScore } from '../modelSelection';

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
        () => new KNearstNeighbors(1),
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
        crossValScore(() => new KNearstNeighbors(1), [[1], [2]], [0], { cv: 2 }),
    ).toThrow('X and y must have the same length');
});
