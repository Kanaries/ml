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

test('manhattan metric uses per-feature median centroids', () => {
    // sklearn semantics: metric='manhattan' computes centroids as the
    // per-feature median, not the mean. Median centroids are 0 and 4.5,
    // so [3.8] is closer to class 1; mean centroids (3.333 and 4.5)
    // would wrongly predict class 0.
    const X = [[0], [0], [10], [4.5], [4.5], [4.5]];
    const y = [0, 0, 0, 1, 1, 1];

    const clf = new NearestCentroid({ metric: 'manhattan' });
    clf.fit(X, y);

    expect(clf.predict([[3.8]])).toEqual([1]);
});

test('manhattan median averages the two middle values for even-sized classes', () => {
    // Class 0 median = median(0, 2, 8, 30) = (2 + 8) / 2 = 5 (numpy.median
    // convention; the mean would be 10). Class 1 centroid = 20.
    // Query 13: |13 - 5| = 8 > |13 - 20| = 7 -> class 1 under median centroids,
    // while mean centroids (10 vs 20) would predict class 0.
    const X = [[0], [2], [8], [30], [20], [20]];
    const y = [0, 0, 0, 0, 1, 1];

    const clf = new NearestCentroid({ metric: 'manhattan' });
    clf.fit(X, y);

    expect(clf.predict([[13]])).toEqual([1]);
});

test('validates predict before fit', () => {
    const clf = new NearestCentroid();
    expect(() => clf.predict([[0, 0]])).toThrow('model is not fitted');
});
