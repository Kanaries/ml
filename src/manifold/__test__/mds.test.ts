import { MDS } from '../mds';

function pairwise(X: number[][]): number[][] {
    const n = X.length;
    const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < i; j++) {
            let s = 0;
            for (let k = 0; k < X[i].length; k++) {
                const d = X[i][k] - X[j][k];
                s += d * d;
            }
            const v = Math.sqrt(s);
            D[i][j] = v;
            D[j][i] = v;
        }
    }
    return D;
}

test('mds preserves distances', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const dist = pairwise(X);
    const mds = new MDS({ nComponents: 2 });
    const Y = mds.fitTransform(X);
    const distY = pairwise(Y);
    for (let i = 0; i < dist.length; i++) {
        for (let j = 0; j < dist.length; j++) {
            expect(distY[i][j]).toBeCloseTo(dist[i][j], 2);
        }
    }
});
