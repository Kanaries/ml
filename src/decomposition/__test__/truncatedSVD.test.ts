import { TruncatedSVD } from '../truncatedSVD';

test('basic truncated svd', () => {
    const X = [
        [1, 2, 3],
        [4, 5, 9],
        [7, 8, 15],
        [2, 1, 3],
        [3, 2, 5]
    ];
    const svd = new TruncatedSVD(2);
    const T = svd.fitTransform(X);
    expect(T.length).toBe(5);
    const Xinv = svd.inverseTransform(T);
    for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[i].length; j++) {
            expect(Xinv[i][j]).toBeCloseTo(X[i][j], 0);
        }
    }
});
