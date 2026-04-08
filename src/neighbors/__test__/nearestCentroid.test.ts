import { NearestCentroid } from '../nearestCentroid';

test('init', () => {
    const clf = new NearestCentroid();
    expect(clf).toBeDefined();
});

test('predicts the nearest class centroid', () => {
    const X = [
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11],
    ];
    const y = [0, 0, 1, 1];

    const clf = new NearestCentroid();
    clf.fit(X, y);

    expect(clf.predict([[0, 0.2], [9.5, 10.8]])).toEqual([0, 1]);
});

test('supports manhattan distance', () => {
    const X = [
        [0, 0],
        [0, 2],
        [5, 5],
        [6, 5],
    ];
    const y = [0, 0, 1, 1];

    const clf = new NearestCentroid({ metric: 'manhattan' });
    clf.fit(X, y);

    expect(clf.predict([[0, 1], [6, 4]])).toEqual([0, 1]);
});

test('validates predict before fit', () => {
    const clf = new NearestCentroid();
    expect(() => clf.predict([[0, 0]])).toThrow('model is not fitted');
});
