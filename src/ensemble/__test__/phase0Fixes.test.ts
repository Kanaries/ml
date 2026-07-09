import { AdaBoostClassifier } from '../adaBoostClassifier';
import { AdaBoostRegressor } from '../adaBoostRegressor';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { DecisionTreeRegressor } from '../../tree/decisionTreeRegressor';

describe('AdaBoostClassifier label handling', () => {
    const X = [[1], [2], [3], [4], [6], [7], [8], [9]];

    test('supports labels other than {0, 1}', () => {
        const y = [1, 1, 1, 1, 2, 2, 2, 2];
        const clf = new AdaBoostClassifier({ nEstimators: 10 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });

    test('supports {-1, 1} labels', () => {
        const y = [-1, -1, -1, -1, 1, 1, 1, 1];
        const clf = new AdaBoostClassifier({ nEstimators: 10 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });

    test('a failed refit does not corrupt a fitted model', () => {
        const y = [5, 5, 5, 5, 9, 9, 9, 9];
        const clf = new AdaBoostClassifier({ nEstimators: 10 });
        clf.fit(X, y);
        expect(() => clf.fit([[1], [2]], [1, 1])).toThrow();
        expect(clf.predict([[1], [9]])).toEqual([5, 9]);
    });

    test('predictProba is ordered by sorted class labels', () => {
        const y = [5, 5, 5, 5, 9, 9, 9, 9];
        const clf = new AdaBoostClassifier({ nEstimators: 10 });
        clf.fit(X, y);
        const proba = clf.predictProba([[1], [9]]);
        expect(proba[0][0]).toBeGreaterThan(proba[0][1]); // class 5 more likely
        expect(proba[1][1]).toBeGreaterThan(proba[1][0]); // class 9 more likely
    });
});

describe('AdaBoostRegressor', () => {
    test('perfectly fittable target keeps the estimator', () => {
        const X = [[1], [2], [3], [4], [5], [6]];
        const y = [10, 10, 10, 10, 10, 10];
        const reg = new AdaBoostRegressor({ n_estimators: 10, randomState: 42 });
        reg.fit(X, y);
        for (const p of reg.predict([[1.5], [3.5]])) {
            expect(p).toBeCloseTo(10, 6);
        }
    });

    test('randomState makes fit reproducible', () => {
        const X = Array.from({ length: 40 }, (_, i) => [i]);
        const y = X.map(([v]) => v * 2 + Math.sin(v));
        const a = new AdaBoostRegressor({ n_estimators: 10, randomState: 7 });
        const b = new AdaBoostRegressor({ n_estimators: 10, randomState: 7 });
        a.fit(X, y);
        b.fit(X, y);
        expect(a.predict(X)).toEqual(b.predict(X));
    });

    test('weak first estimator with error >= 0.5 is kept, not dropped', () => {
        // depth-0 trees predict a constant, so the first round's weighted
        // error exceeds 0.5 — the ensemble must not end up empty
        const X = Array.from({ length: 10 }, (_, i) => [i]);
        const y = X.map((_, i) => (i < 6 ? 0 : 1000));
        const reg = new AdaBoostRegressor({
            estimator: new DecisionTreeRegressor({ max_depth: 0 }),
            n_estimators: 1,
            randomState: 0,
        });
        reg.fit(X, y);
        expect((reg as any).estimators.length).toBeGreaterThan(0);
        // constant estimator predicts the (resampled) mean, far from 0
        expect(reg.predict([[0]])[0]).toBeGreaterThan(0);
    });

    test('fits a simple linear target reasonably', () => {
        const X = Array.from({ length: 50 }, (_, i) => [i]);
        const y = X.map(([v]) => 3 * v + 1);
        const reg = new AdaBoostRegressor({ n_estimators: 20, randomState: 0 });
        reg.fit(X, y);
        const pred = reg.predict(X);
        const mse = pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / pred.length;
        expect(mse).toBeLessThan(50);
    });
});

describe('DecisionTreeClassifier edge cases', () => {
    test('identical rows with conflicting labels predict the majority class', () => {
        const X = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]];
        const y = [7, 7, 7, 3, 3];
        const clf = new DecisionTreeClassifier();
        clf.fit(X, y);
        expect(clf.predict([[1, 1]])).toEqual([7]);
    });

    test('max_depth=1 builds a stump (at most 2 distinct leaf values)', () => {
        const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
        const y = [0, 0, 1, 1, 0, 0, 1, 1];
        const clf = new DecisionTreeClassifier({ max_depth: 1 });
        clf.fit(X, y);
        const tree = (clf as any).dtree;
        const depth = (t: any): number => (t === null ? -1 : 1 + Math.max(depth(t.leftChild), depth(t.rightChild)));
        expect(depth(tree)).toBeLessThanOrEqual(1);
    });

    test('learns XOR (zero-gain first split must still happen)', () => {
        const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const y = [0, 1, 1, 0];
        const clf = new DecisionTreeClassifier();
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });
});

describe('DecisionTreeRegressor fixes', () => {
    test('max_depth is enforced while building', () => {
        const n = 200;
        const X = Array.from({ length: n }, (_, i) => [i]);
        const y = X.map(([v]) => v + Math.sin(v));
        const reg = new DecisionTreeRegressor({ max_depth: 2 });
        reg.fit(X, y);
        const tree = (reg as any).regTree;
        const depth = (t: any): number => (t === null ? -1 : 1 + Math.max(depth(t.leftChild), depth(t.rightChild)));
        expect(depth(tree)).toBeLessThanOrEqual(2);
    });

    test('max_depth=1 yields at most 2 distinct predictions', () => {
        const X = Array.from({ length: 20 }, (_, i) => [i]);
        const y = X.map(([v]) => v);
        const reg = new DecisionTreeRegressor({ max_depth: 1 });
        reg.fit(X, y);
        expect(new Set(reg.predict(X)).size).toBeLessThanOrEqual(2);
    });

    test('split criterion weights children by sample count', () => {
        // Weighted SSE prefers isolating the single outlier at x=10;
        // the old unweighted variance-sum could pick a worse split.
        const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
        const y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 100];
        const reg = new DecisionTreeRegressor({ max_depth: 1 });
        reg.fit(X, y);
        const pred = reg.predict(X);
        expect(pred.slice(0, 9)).toEqual(new Array(9).fill(0));
        expect(pred[9]).toBeCloseTo(100, 6);
    });

    test('huge feature values do not overflow the midpoint threshold', () => {
        const X = [[1e308], [1.1e308], [1.2e308], [1.3e308]];
        const y = [0, 0, 10, 10];
        const reg = new DecisionTreeRegressor({ max_depth: 1 });
        reg.fit(X, y);
        expect(reg.predict(X)).toEqual([0, 0, 10, 10]);
    });

    test('constant target predicts the constant', () => {
        const X = [[1], [2], [3]];
        const y = [5, 5, 5];
        const reg = new DecisionTreeRegressor();
        reg.fit(X, y);
        expect(reg.predict([[2]])).toEqual([5]);
    });
});
