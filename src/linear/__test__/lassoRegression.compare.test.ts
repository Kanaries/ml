import { LassoRegression } from '../lassoRegression';

test('compare with sklearn', () => {
    const trainX = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 1]
    ];
    const trainY = [1, 3, 0, 2, 4, -1, 1, 6];
    const testX = [
        [1.5, 1.5],
        [2.5, 1],
        [0.5, 2],
        [3, 2]
    ];
    const expected = [1.6, 4.46153846, -1.26153846, 3.66153846];

    const lasso = new LassoRegression({
        alpha: 0.1,
        fitIntercept: true,
        maxIter: 20000,
        tol: 1e-12
    });
    lasso.fit(trainX, trainY);
    const pred = lasso.predict(testX);
    expect(pred.length).toBe(expected.length);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(expected[i], 5);
    }
});
