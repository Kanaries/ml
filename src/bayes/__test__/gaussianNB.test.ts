import { GaussianNB } from '../gaussianNB';

test('init', () => {
    const nb = new GaussianNB();
    expect(nb).toBeDefined();
});

test('GaussianNB matches sklearn reference predictions on a separable dataset', () => {
    const X = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [2, 1],
        [1, 2],
    ];
    const y = [0, 0, 0, 1, 1, 1];
    const testX = [
        [-1.5, -1.5],
        [1.5, 1.5],
        [0, 0],
    ];

    const nb = new GaussianNB();
    nb.fit(X, y);

    expect(nb.predict(testX)).toEqual([0, 1, 0]);
});
