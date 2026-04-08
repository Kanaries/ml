import { RandomForestRegressor } from '../randomForestRegressor';

test('RandomForestRegressor initializes', () => {
    expect(new RandomForestRegressor()).toBeDefined();
});

test('RandomForestRegressor fits and predicts a simple regression dataset', () => {
    const X = [[0], [1], [2], [3], [4], [5]];
    const y = [0, 2, 4, 6, 8, 10];

    const reg = new RandomForestRegressor({ nEstimators: 15, randomState: 42 });
    reg.fit(X, y);

    expect(reg.predict([[1.5]])[0]).toBeGreaterThan(0);
    expect(reg.predict([[1.5]])[0]).toBeLessThan(6);
});
