import { PCA } from '../pca';

test('basic pca', () => {
    const X = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 1],
        [2, 4, 5],
        [3, 6, 0]
    ];
    const pca = new PCA(2);
    const T = pca.fitTransform(X);
    expect(T.length).toBe(5);
    const Xinv = pca.inverseTransform(T);
    expect(Xinv.length).toBe(5);
    for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[i].length; j++) {
            expect(Xinv[i][j]).toBeCloseTo(X[i][j], 0);
        }
    }
});
