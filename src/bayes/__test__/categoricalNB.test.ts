import { CategoricalNB } from '../categoricalNB';

test('init', () => {
    const nb = new CategoricalNB();
    expect(nb).toBeDefined();
});

test('simple classification', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const Y = [0, 0, 1, 1];
    const nb = new CategoricalNB();
    nb.fit(X, Y);
    const pred = nb.predict(X);
    expect(pred).toEqual(Y);
});
