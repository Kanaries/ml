import { KNeighborsRegressor } from '../kneighborsRegressor';

test('init', () => {
    const reg = new KNeighborsRegressor();
    expect(reg).toBeDefined();
});

test('predicts the mean of the nearest neighbors', () => {
    const X = [
        [0],
        [1],
        [2],
        [3],
    ];
    const y = [0, 1, 2, 3];

    const reg = new KNeighborsRegressor({ nNeighbors: 2 });
    reg.fit(X, y);

    expect(reg.predict([[1.1], [2.6]])).toEqual([1.5, 2.5]);
});

test('supports distance weighting', () => {
    const X = [
        [0],
        [1],
        [2],
    ];
    const y = [0, 10, 20];

    const reg = new KNeighborsRegressor({ nNeighbors: 2, weights: 'distance' });
    reg.fit(X, y);

    expect(reg.predict([[1.25]])[0]).toBeCloseTo((10 / 0.25 + 20 / 0.75) / (1 / 0.25 + 1 / 0.75), 8);
});

test('validates fit and predict calls', () => {
    const reg = new KNeighborsRegressor();
    expect(() => reg.predict([[0]])).toThrow('model is not fitted');
    expect(() => reg.fit([[0], [1]], [1])).toThrow('X and Y must have the same length');
});
