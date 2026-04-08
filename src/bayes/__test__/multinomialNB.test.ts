import { MultinomialNB } from '../multinomialNB';

test('init', () => {
    const nb = new MultinomialNB();
    expect(nb).toBeDefined();
});

test('MultinomialNB matches sklearn reference predictions on count data', () => {
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

    const nb = new MultinomialNB();
    nb.fit(X, y);

    expect(nb.predict(testX)).toEqual([0, 1, 1]);
});

test('MultinomialNB rejects negative feature values', () => {
    const nb = new MultinomialNB();
    expect(() => nb.fit([[1, -1]], [0])).toThrow('MultinomialNB requires non-negative feature values');
});
