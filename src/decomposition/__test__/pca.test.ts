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

test('pca stays finite on exactly anti-correlated features', () => {
    // (1, 1) is an exact eigenvector of this covariance matrix, which used to
    // stall the deterministic all-ones power iteration and produce NaN.
    const X = [
        [1, -1],
        [2, -2],
        [3, -3],
        [4, -4],
        [-1, 1],
        [-2, 2]
    ];
    const pca = new PCA(2);
    pca.fit(X);
    const comps = pca.getComponents();
    expect(comps.length).toBe(2);
    for (const row of comps) {
        for (const v of row) {
            expect(Number.isFinite(v)).toBe(true);
        }
    }
    // first principal component must align with (1, -1)/sqrt(2) up to sign
    const first = comps[0];
    const alignment = Math.abs(first[0] * Math.SQRT1_2 - first[1] * Math.SQRT1_2);
    expect(alignment).toBeCloseTo(1, 6);
    for (const ev of pca.getExplainedVariance()) {
        expect(Number.isFinite(ev)).toBe(true);
        expect(ev).toBeGreaterThanOrEqual(0);
    }
});

test('pca returns orthogonal components on a near-degenerate spectrum', () => {
    // covariance is approximately diag(0.5, 0.5 * 0.9999^2): two eigenvalues
    // so close that unconverged power iteration mixes the components unless
    // each extracted vector is re-orthogonalized against the previous ones.
    const X: number[][] = [];
    for (let i = 0; i < 40; i++) {
        const t = (i / 40) * 2 * Math.PI;
        X.push([Math.cos(t), 0.9999 * Math.sin(t)]);
    }
    const pca = new PCA(2);
    pca.fit(X);
    const comps = pca.getComponents();
    expect(comps.length).toBe(2);
    for (const row of comps) {
        for (const v of row) {
            expect(Number.isFinite(v)).toBe(true);
        }
    }
    const d = comps[0][0] * comps[1][0] + comps[0][1] * comps[1][1];
    expect(Math.abs(d)).toBeLessThan(1e-6);
});
