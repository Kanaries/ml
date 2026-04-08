import { RadiusNeighborsRegressor } from '../radiusNeighborsRegressor';

test('init', () => {
    const reg = new RadiusNeighborsRegressor();
    expect(reg).toBeDefined();
});

test('predicts the mean of neighbors inside the radius', () => {
    const X = [
        [0],
        [1],
        [2],
        [3],
    ];
    const y = [0, 2, 4, 6];

    const reg = new RadiusNeighborsRegressor({ radius: 1.1 });
    reg.fit(X, y);

    expect(reg.predict([[1.0]])).toEqual([2]);
});

test('supports distance weighting', () => {
    const X = [
        [0],
        [1],
        [2],
    ];
    const y = [0, 10, 20];

    const reg = new RadiusNeighborsRegressor({ radius: 1.1, weights: 'distance' });
    reg.fit(X, y);

    expect(reg.predict([[1.25]])[0]).toBeCloseTo((10 / 0.25 + 20 / 0.75) / (1 / 0.25 + 1 / 0.75), 8);
});

test('returns NaN when no neighbors are found', () => {
    const reg = new RadiusNeighborsRegressor({ radius: 0.1 });
    reg.fit([[0]], [1]);
    expect(Number.isNaN(reg.predict([[10]])[0])).toBe(true);
});
