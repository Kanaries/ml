import { LocallyLinearEmbedding } from '../lle';

test('basic lle', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const lle = new LocallyLinearEmbedding(2, 2);
    const Y = lle.fitTransform(X);
    expect(Y.length).toBe(4);
    const T = lle.transform([[0.5, 0.5]]);
    expect(T.length).toBe(1);
});
