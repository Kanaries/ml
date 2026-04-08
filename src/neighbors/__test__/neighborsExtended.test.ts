import { KNeighborsRegressor } from '../kneighborsRegressor';
import { RadiusNeighborsClassifier } from '../radiusNeighborsClassifier';
import { RadiusNeighborsRegressor } from '../radiusNeighborsRegressor';
import { NearestCentroid } from '../nearestCentroid';

test('KNeighborsRegressor matches sklearn reference outputs', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 1, 2, 3];
    const testX = [[1.5], [2.5]];

    const uniform = new KNeighborsRegressor({ nNeighbors: 2, weights: 'uniform' });
    uniform.fit(X, y);
    expect(uniform.predict(testX)).toEqual([1.5, 2.5]);

    const distance = new KNeighborsRegressor({ nNeighbors: 2, weights: 'distance' });
    distance.fit(X, y);
    expect(distance.predict(testX)).toEqual([1.5, 2.5]);
});

test('RadiusNeighborsClassifier matches sklearn reference outputs and supports outlier labels', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 0, 1, 1];
    const testX = [[0.9], [2.1], [10]];

    const clf = new RadiusNeighborsClassifier({ radius: 1.0, outlierLabel: -1 });
    clf.fit(X, y);

    expect(clf.predict(testX)).toEqual([0, 1, -1]);
});

test('RadiusNeighborsClassifier throws when no neighbors are found and no outlier label is configured', () => {
    const clf = new RadiusNeighborsClassifier({ radius: 0.5 });
    clf.fit([[0], [1]], [0, 1]);

    expect(() => clf.predict([[10]])).toThrow('No neighbors found within the configured radius');
});

test('RadiusNeighborsRegressor matches sklearn reference outputs and returns NaN for empty neighborhoods', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 1, 2, 3];

    const reg = new RadiusNeighborsRegressor({ radius: 1.0 });
    reg.fit(X, y);

    expect(reg.predict([[0.5], [2.5]])).toEqual([0.5, 2.5]);
    expect(Number.isNaN(reg.predict([[10]])[0])).toBe(true);
});

test('NearestCentroid matches sklearn reference outputs', () => {
    const X = [
        [1, 2],
        [1, 4],
        [4, 2],
        [4, 4],
    ];
    const y = [0, 0, 1, 1];
    const testX = [
        [1, 3],
        [4, 3],
        [2.5, 3],
    ];

    const clf = new NearestCentroid();
    clf.fit(X, y);

    expect(clf.predict(testX)).toEqual([0, 1, 0]);
});
