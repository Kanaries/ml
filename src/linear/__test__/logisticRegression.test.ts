import { LogisticRegression } from '../logisticRegression';

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

describe('label handling (sklearn-style classes_ mapping)', () => {
    test('arbitrary binary labels are learned and returned as-is', () => {
        const X = [[0], [1], [2], [3], [10], [11], [12], [13]];
        const Y = [1, 1, 1, 1, 2, 2, 2, 2];
        const m = new LogisticRegression({ learningRate: 0.1, maxIter: 2000 });
        m.fit(X, Y);
        expect(m.predict([[0], [13]])).toEqual([1, 2]);
    });

    test('more than two classes throws', () => {
        const m = new LogisticRegression();
        expect(() => m.fit([[0], [1], [2]], [0, 1, 2])).toThrow();
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
