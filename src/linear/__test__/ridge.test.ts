import { Ridge } from '../ridge';
import { RidgeRegression } from '../ridgeRegression';

test('Ridge is an sklearn-style alias of RidgeRegression', () => {
    const X = [[0], [1], [2], [3]];
    const y = [1, 3, 5, 7];

    const ridge = new Ridge({ alpha: 0.5 });
    const regression = new RidgeRegression({ alpha: 0.5 });

    ridge.fit(X, y);
    regression.fit(X, y);

    expect(ridge.predict([[4], [5]])).toEqual(regression.predict([[4], [5]]));
});

test('Ridge matches the existing ridge regression sklearn reference values', () => {
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
    const expected = [1.688311688312, 4.207792207792, -0.831168831169, 3.584415584416];

    const ridge = new Ridge({ alpha: 1, fitIntercept: true });
    ridge.fit(trainX, trainY);
    const pred = ridge.predict(testX);

    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(expected[i], 6);
    }
});
