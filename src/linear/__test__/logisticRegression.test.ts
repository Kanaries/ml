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
