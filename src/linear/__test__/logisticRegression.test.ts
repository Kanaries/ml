import { LogisticRegression } from '../logisticRegression';
import { loadModel } from '../../base';

test('init', () => {
    const lr = new LogisticRegression();
    expect(lr).toBeDefined();
});

test('simple classification', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, 1, 1];
    const lr = new LogisticRegression({ learningRate: 0.5, maxIter: 200 });
    lr.fit(X, Y);
    const pred = lr.predict(X);
    expect(pred).toEqual(Y);
});

test('C controls inverse L2 regularization strength', () => {
    const X = [[-3], [-2], [-1], [1], [2], [3]];
    const y = [0, 0, 0, 1, 1, 1];
    const strong = new LogisticRegression({ learningRate: 0.2, maxIter: 1000, C: 0.01 });
    const weak = new LogisticRegression({ learningRate: 0.2, maxIter: 1000, C: 100 });
    strong.fit(X, y);
    weak.fit(X, y);
    expect(Math.abs(strong.coef[0] as number)).toBeLessThan(Math.abs(weak.coef[0] as number));
    expect(() => new LogisticRegression({ C: 0 })).toThrow(/C/);
    expect(() => new LogisticRegression().setParams({ C: 0 }).fit(X, y)).toThrow(/C|parameters/);
});

describe('label handling (sklearn-style classes_ mapping)', () => {
    test('arbitrary binary labels are learned and returned as-is', () => {
        const X = [[0], [1], [2], [3], [10], [11], [12], [13]];
        const Y = [1, 1, 1, 1, 2, 2, 2, 2];
        const m = new LogisticRegression({ learningRate: 0.1, maxIter: 2000 });
        m.fit(X, Y);
        expect(m.predict([[0], [13]])).toEqual([1, 2]);
    });

    test('multiclass targets use multinomial scores and expose a coefficient row per class', () => {
        const m = new LogisticRegression({ learningRate: 0.2, maxIter: 1000, C: 1 });
        const X = [[0, 0], [0, 1], [5, 0], [6, 0], [0, 5], [0, 6]];
        const y = [0, 0, 1, 1, 2, 2];
        m.fit(X, y);
        expect(m.predict(X)).toEqual(y);
        expect(m.predictProba(X).every(row => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12)).toBe(true);
        expect(m.coef).toHaveLength(3);
        const revived = loadModel(JSON.stringify(m)) as LogisticRegression;
        expect(revived.predict(X)).toEqual(y);
        expect(revived.predictProba(X)).toEqual(m.predictProba(X));
    });

    test('empty input throws', () => {
        const m = new LogisticRegression();
        expect(() => m.fit([], [])).toThrow();
    });

    test('predict before fit throws', () => {
        const m = new LogisticRegression();
        expect(() => m.predict([[1]])).toThrow();
    });
});
