import { ComplementNB } from '../complementNB';

test('init', () => {
    const nb = new ComplementNB();
    expect(nb).toBeDefined();
});

test('ComplementNB matches sklearn reference predictions on count data', () => {
    const X = [
        [2, 1, 0],
        [1, 0, 1],
        [0, 2, 3],
        [0, 1, 2],
    ];
    const y = [0, 0, 1, 1];
    const testX = [
        [1, 0, 0],
        [0, 1, 2],
        [0, 0, 4],
    ];

    const nb = new ComplementNB();
    nb.fit(X, y);

    expect(nb.predict(testX)).toEqual([0, 1, 1]);
});

test('ComplementNB rejects negative feature values', () => {
    const nb = new ComplementNB();
    expect(() => nb.fit([[1, -1]], [0])).toThrow('ComplementNB requires non-negative feature values');
});
