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

test('truncated svd stays finite on exactly complementary columns', () => {
    // (1, 1) is an exact eigenvector of X^T X here, which used to stall the
    // deterministic all-ones power iteration and produce NaN everywhere.
    const X = [
        [1, -1],
        [2, -2],
        [3, -3],
        [-1, 1],
        [-2, 2]
    ];
    const svd = new TruncatedSVD(2);
    const T = svd.fitTransform(X);
    for (const row of svd.getComponents()) {
        for (const v of row) {
            expect(Number.isFinite(v)).toBe(true);
        }
    }
    for (const s of svd.getSingularValues()) {
        expect(Number.isFinite(s)).toBe(true);
        expect(s).toBeGreaterThanOrEqual(0);
    }
    for (const ev of svd.getExplainedVariance()) {
        expect(Number.isFinite(ev)).toBe(true);
        expect(ev).toBeGreaterThanOrEqual(0);
    }
    for (const row of T) {
        for (const v of row) {
            expect(Number.isFinite(v)).toBe(true);
        }
    }
});

describe('explained variance (sklearn definition)', () => {
    test('ratio stays within (0, 1] on data with a large mean', () => {
        // deterministic 20x3 data with mean ~100 (typical count-matrix scale)
        const X: number[][] = [];
        for (let i = 0; i < 20; i++) {
            X.push([100 + Math.sin(i), 100 + Math.cos(i * 1.3) * 2, 100 + Math.sin(i * 0.7) * 3]);
        }
        const svd = new TruncatedSVD(2);
        const T = svd.fitTransform(X);
        const ratios = svd.getExplainedVarianceRatio();
        for (const r of ratios) {
            expect(r).toBeGreaterThanOrEqual(0);
            expect(r).toBeLessThanOrEqual(1 + 1e-9);
        }
        expect(ratios.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1 + 1e-9);

        // explainedVariance[i] must equal Var(X_transformed[:, i]) (ddof=0)
        const vars = svd.getExplainedVariance();
        for (let c = 0; c < 2; c++) {
            const col = T.map(row => row[c]);
            const mu = col.reduce((a, b) => a + b, 0) / col.length;
            const v = col.reduce((a, b) => a + (b - mu) ** 2, 0) / col.length;
            expect(vars[c]).toBeCloseTo(v, 6);
        }
    });
});
