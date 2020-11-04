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
    console.log('tree', dtree)
    expect(ans).toEqual([0.5]);
});
