import { RadiusNeighborsClassifier } from '../radiusNeighborsClassifier';

test('init', () => {
    const clf = new RadiusNeighborsClassifier();
    expect(clf).toBeDefined();
});

test('predicts by majority vote inside the radius', () => {
    const X = [
        [0],
        [1],
        [2],
        [3],
    ];
    const y = [0, 0, 1, 1];

    const clf = new RadiusNeighborsClassifier({ radius: 0.75 });
    clf.fit(X, y);

    expect(clf.predict([[0.2], [2.2]])).toEqual([0, 1]);
});

test('supports distance weighting', () => {
    const X = [
        [0],
        [1],
        [2],
    ];
    const y = [0, 1, 1];

    const clf = new RadiusNeighborsClassifier({ radius: 1.1, weights: 'distance' });
    clf.fit(X, y);

    expect(clf.predict([[1.0]])).toEqual([1]);
});

test('throws when no neighbors are found', () => {
    const clf = new RadiusNeighborsClassifier({ radius: 0.1 });
    clf.fit([[0]], [1]);
    expect(() => clf.predict([[10]])).toThrow('No neighbors found within the configured radius');
});
