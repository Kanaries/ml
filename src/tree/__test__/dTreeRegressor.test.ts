import { DecisionTreeRegressor } from '../decisionTreeRegressor';

test('dtree init', () => {
    const dtree = new DecisionTreeRegressor();
    expect(dtree).toBeDefined();
});

test('basic case', () => {
    const X = [
        [0, 0],
        [2, 2],
        [3, 6]
    ];
    const Y = [0.5, 2.5, 3.6];
    const dtree = new DecisionTreeRegressor();
    dtree.fit(X, Y);
    const ans = dtree.predict([[1, 1]]);
    expect(ans).toEqual([0.5]);
});

test('case 1', () => {
    const trainX = [
        [6, 4, 0],
        [4, 0, 5],
        [6, 2, 0],
        [8, 3, 0],
        [9, 2, 2],
        [1, 1, 5],
        [2, 2, 4],
        [9, 10, 1],
        [7, 6, 3],
        [5, 6, 4],
        [8, 2, 4],
        [6, 0, 5],
        [1, 9, 1],
        [3, 0, 4],
        [4, 5, 0],
        [4, 7, 2],
        [3, 1, 3],
        [9, 8, 0],
        [2, 7, 0],
        [4, 8, 4],
    ];
    const trainY = [140, -20, 100, 140, 106, -30, 12, 278, 154, 122, 72, 0, 178, -18, 140, 156, 14, 250, 160, 152];
    const testX = [
        [9, 4, 1],
        [3, 0, 4],
        [5, 5, 2],
        [7, 2, 5],
        [7, 3, 3],
        [4, 2, 5],
        [3, 3, 1],
        [2, 4, 2],
        [1, 7, 3],
        [6, 8, 2],
    ];
    const testY = [250, -18, 154,  72, 154,  12, 140, 154, 178, 156];
    const regTree = new DecisionTreeRegressor();
    regTree.fit(trainX, trainY);
    const result = regTree.predict(testX);
    console.log('result', result)
    console.log('testY', testY)
    expect(result).toEqual(testY)
})
