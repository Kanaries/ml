import { Lasso } from '../lasso';
import { LassoRegression } from '../lassoRegression';

test('Lasso is an sklearn-style alias of LassoRegression', () => {
    const X = [[0], [1], [2], [3]];
    const y = [1, 3, 5, 7];

    const lasso = new Lasso({ alpha: 0.1, maxIter: 5000, tol: 1e-10 });
    const regression = new LassoRegression({ alpha: 0.1, maxIter: 5000, tol: 1e-10 });

    lasso.fit(X, y);
    regression.fit(X, y);

    const ridgePred = lasso.predict([[4], [5]]);
    const regressionPred = regression.predict([[4], [5]]);
    expect(ridgePred[0]).toBeCloseTo(regressionPred[0], 8);
    expect(ridgePred[1]).toBeCloseTo(regressionPred[1], 8);
});

test('Lasso matches the existing lasso sklearn reference values', () => {
    const trainX = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 1],
    ];
    const trainY = [1, 3, 0, 2, 4, -1, 1, 6];
    const testX = [
        [1.5, 1.5],
        [2.5, 1],
        [0.5, 2],
        [3, 2],
    ];
    const expected = [1.6, 4.46153846, -1.26153846, 3.66153846];

    const lasso = new Lasso({
        alpha: 0.1,
        fitIntercept: true,
        maxIter: 20000,
        tol: 1e-12,
    });
    lasso.fit(trainX, trainY);
    const pred = lasso.predict(testX);

    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(expected[i], 5);
    }
});
