import { AdaBoostClassifier } from '../adaBoostClassifier';

test('init', () => {
    const clf = new AdaBoostClassifier();
    expect(clf).toBeDefined();
});

test('simple classification', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, 1, 1];
    const clf = new AdaBoostClassifier({ nEstimators: 5 });
    clf.fit(X, Y);
    const pred = clf.predict(X);
    expect(pred).toEqual(Y);
});

test('supports snake_case constructor aliases', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, 1, 1];
    const clf = new AdaBoostClassifier({ n_estimators: 5, learning_rate: 0.5 });
    clf.fit(X, Y);
    const pred = clf.predict(X);
    expect(pred).toEqual(Y);
});

test('getFeatureImportances returns one entry per input feature', () => {
    // 10-dimensional data where only feature 0 is informative: the stumps
    // never use features 1..9, but the importance vector must still cover
    // all 10 input features
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < 20; i++) {
        const row = new Array(10).fill(0);
        row[0] = i < 10 ? i : i + 10;
        row[1] = i % 2;
        X.push(row);
        Y.push(i < 10 ? 0 : 1);
    }
    const clf = new AdaBoostClassifier({ nEstimators: 10 });
    clf.fit(X, Y);
    const importances = clf.getFeatureImportances();
    expect(importances).toHaveLength(10);
    expect(importances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
});

test('binary weight update follows exact SAMME (no ad-hoc weight cap)', () => {
    // 102 points with one flipped label deep inside the negative side: the
    // first stump (threshold 50.5) has error 1/102, so exp(alpha) = 101 and
    // a cap at 100x the max weight would distort the second round.
    // Exact SAMME: after renormalization the flipped point carries weight
    // 101/202, so the second stump (threshold 24.5) has error 25/202 and
    // alpha = log(177/25). With the cap the weight is 100/201 and the alpha
    // would be log(176/25) instead.
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i <= 101; i++) {
        X.push([i]);
        Y.push(i >= 51 ? 1 : 0);
    }
    Y[25] = 1;
    const clf = new AdaBoostClassifier({ nEstimators: 3 });
    clf.fit(X, Y);
    const alphas = (clf as any).estimatorWeights as number[];
    expect(alphas[0]).toBeCloseTo(Math.log(101), 9);
    expect(alphas[1]).toBeCloseTo(Math.log(177 / 25), 9);
});
