import { LinearRegression } from '../../linear';
import { crossValPredict, KFold } from '../modelSelection';
import { KNearestNeighbors } from '../../neighbors/knn';
import { Pipeline } from '../../pipeline';
import { StandardScaler } from '../preprocessing';

test('crossValPredict returns each out-of-fold prediction in original order', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i]);
    const y = X.map(row => 2 * row[0] + 1);
    const predictions = crossValPredict(() => new LinearRegression(), X, y, { cv: new KFold({ nSplits: 3 }) });
    predictions.forEach((value, i) => expect(value).toBeCloseTo(y[i], 10));
});

test('crossValPredict restores nonsequential test folds to original order', () => {
    class MeanEstimator {
        private value = 0;
        public fit(_X: number[][], y: number[]): void { this.value = y.reduce((a, b) => a + b, 0) / y.length; }
        public predict(X: number[][]): number[] { return new Array(X.length).fill(this.value); }
    }
    const X = Array.from({ length: 6 }, (_, i) => [i]);
    const splitter = { split: () => [
        { trainIndices: [0, 1, 3, 4], testIndices: [2, 5] },
        { trainIndices: [1, 2, 4, 5], testIndices: [0, 3] },
        { trainIndices: [0, 2, 3, 5], testIndices: [1, 4] },
    ] };
    expect(crossValPredict(() => new MeanEstimator(), X, [0, 1, 2, 3, 4, 5], { cv: splitter }))
        .toEqual([3, 2.5, 2, 3, 2.5, 2]);
});

test('crossValPredict defaults to stratified folds for classifiers', () => {
    const X = [[0], [.1], [10], [10.1]];
    const y = [0, 0, 1, 1];
    expect(crossValPredict(() => new KNearestNeighbors({ kNeighbors: 1 }), X, y, { cv: 2 })).toEqual(y);
});

test('crossValPredict recognizes a classifier in the final Pipeline step', () => {
    const X = [[0], [.1], [10], [10.1]];
    const y = [0, 0, 1, 1];
    const factory = () => new Pipeline({ steps: [
        ['scale', new StandardScaler()],
        ['classifier', new KNearestNeighbors({ kNeighbors: 1 })],
    ] });
    expect(crossValPredict(factory, X, y, { cv: 2 })).toEqual(y);
});
