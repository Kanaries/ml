import { RidgeRegression } from '../ridgeRegression';

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
    const expected = [1.688311688312, 4.207792207792, -0.831168831169, 3.584415584416];

    const rr = new RidgeRegression({ alpha: 1, fitIntercept: true });
    rr.fit(trainX, trainY);
    const pred = rr.predict(testX);
    expect(pred.length).toBe(expected.length);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(expected[i], 6);
    }
});
