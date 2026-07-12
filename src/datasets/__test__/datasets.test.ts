import { makeBlobs } from '../makeBlobs';
import { makeClassification } from '../makeClassification';
import { makeRegression } from '../makeRegression';
import { makeMoons } from '../makeMoons';
import { makeCircles } from '../makeCircles';
import { createRandomGenerator, createGaussianGenerator } from '../common';

function mean(values: number[]): number {
    return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function columnMean(X: number[][], col: number): number {
    return mean(X.map((row) => row[col]));
}

function countLabels(y: number[]): Map<number, number> {
    const counts = new Map<number, number>();
    for (const label of y) {
        counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return counts;
}

describe('createGaussianGenerator', () => {
    test('produces roughly standard normal values', () => {
        const gaussian = createGaussianGenerator(createRandomGenerator(42));
        const values = Array.from({ length: 20000 }, () => gaussian());
        const m = mean(values);
        const variance = mean(values.map((v) => (v - m) * (v - m)));
        expect(Math.abs(m)).toBeLessThan(0.05);
        expect(Math.abs(variance - 1)).toBeLessThan(0.05);
    });
});

describe('makeBlobs', () => {
    test('same seed gives identical output, different seed differs', () => {
        const a = makeBlobs({ nSamples: 60, randomState: 7 });
        const b = makeBlobs({ nSamples: 60, randomState: 7 });
        const c = makeBlobs({ nSamples: 60, randomState: 8 });
        expect(a).toEqual(b);
        expect(a.X).not.toEqual(c.X);
    });

    test('shapes and default label distribution', () => {
        const { X, y } = makeBlobs({ nSamples: 100, nFeatures: 4, centers: 5, randomState: 0 });
        expect(X).toHaveLength(100);
        expect(X[0]).toHaveLength(4);
        expect(y).toHaveLength(100);
        const counts = countLabels(y);
        expect(counts.size).toBe(5);
        for (const count of counts.values()) {
            expect(count).toBe(20);
        }
    });

    test('per-cluster nSamples array controls cluster sizes', () => {
        const { X, y } = makeBlobs({ nSamples: [10, 30, 60], randomState: 3 });
        expect(X).toHaveLength(100);
        const counts = countLabels(y);
        expect(counts.get(0)).toBe(10);
        expect(counts.get(1)).toBe(30);
        expect(counts.get(2)).toBe(60);
    });

    test('cluster means are close to explicit centers', () => {
        const centers = [[-10, -10], [0, 0], [10, 10]];
        const { X, y } = makeBlobs({ nSamples: 600, centers, clusterStd: 0.5, randomState: 11 });
        for (let k = 0; k < centers.length; k++) {
            const members = X.filter((_, i) => y[i] === k);
            expect(members.length).toBe(200);
            for (let d = 0; d < 2; d++) {
                const centroid = mean(members.map((p) => p[d]));
                expect(Math.abs(centroid - centers[k][d])).toBeLessThan(0.2);
            }
        }
    });

    test('per-cluster clusterStd is respected', () => {
        const centers = [[0, 0], [100, 100]];
        const { X, y } = makeBlobs({
            nSamples: 1000, centers, clusterStd: [0.5, 5], shuffle: false, randomState: 5,
        });
        const spread = (k: number) => {
            const members = X.filter((_, i) => y[i] === k);
            const c = centers[k];
            return Math.sqrt(mean(members.map((p) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)) / 2);
        };
        expect(spread(0)).toBeGreaterThan(0.35);
        expect(spread(0)).toBeLessThan(0.65);
        expect(spread(1)).toBeGreaterThan(3.5);
        expect(spread(1)).toBeLessThan(6.5);
    });

    test('shuffle=false keeps samples grouped by cluster', () => {
        const { y } = makeBlobs({ nSamples: 30, centers: 3, shuffle: false, randomState: 1 });
        expect(y).toEqual([...Array(10).fill(0), ...Array(10).fill(1), ...Array(10).fill(2)]);
    });

    test('random centers stay inside centerBox', () => {
        const { X } = makeBlobs({
            nSamples: 300, centers: 3, centerBox: [-1, 1], clusterStd: 0.01, randomState: 2,
        });
        for (const row of X) {
            for (const value of row) {
                expect(value).toBeGreaterThan(-1.2);
                expect(value).toBeLessThan(1.2);
            }
        }
    });

    test('validation errors', () => {
        expect(() => makeBlobs({ nSamples: [10, 10], centers: 2 })).toThrow(/centers must be an array/);
        expect(() => makeBlobs({ nSamples: [10, 10], centers: [[0, 0]] })).toThrow(/centers length/);
        expect(() => makeBlobs({ nSamples: 10, centers: 2, clusterStd: [1, 2, 3] })).toThrow(/clusterStd length/);
    });
});

describe('makeClassification', () => {
    test('same seed gives identical output, different seed differs', () => {
        const a = makeClassification({ randomState: 42 });
        const b = makeClassification({ randomState: 42 });
        const c = makeClassification({ randomState: 43 });
        expect(a).toEqual(b);
        expect(a.X).not.toEqual(c.X);
    });

    test('shapes and default label range', () => {
        const { X, y } = makeClassification({ nSamples: 120, randomState: 0 });
        expect(X).toHaveLength(120);
        expect(X[0]).toHaveLength(20);
        expect(y).toHaveLength(120);
        for (const label of y) {
            expect([0, 1]).toContain(label);
        }
    });

    test('classes are balanced by default (flipY=0)', () => {
        const { y } = makeClassification({
            nSamples: 200, nClasses: 4, nInformative: 5, flipY: 0, randomState: 9,
        });
        const counts = countLabels(y);
        expect(counts.size).toBe(4);
        for (const count of counts.values()) {
            expect(count).toBe(50);
        }
    });

    test('weights control the class distribution', () => {
        const { y } = makeClassification({
            nSamples: 1000, weights: [0.9, 0.1], flipY: 0, randomState: 3,
        });
        const counts = countLabels(y);
        expect(counts.get(0)).toBeGreaterThanOrEqual(890);
        expect(counts.get(0)).toBeLessThanOrEqual(910);
    });

    test('weights of length nClasses - 1 get the last weight inferred', () => {
        const { y } = makeClassification({
            nSamples: 1000, nClasses: 3, nInformative: 4, weights: [0.2, 0.2], flipY: 0, randomState: 3,
        });
        const counts = countLabels(y);
        expect(counts.get(2)).toBeGreaterThanOrEqual(590);
        expect(counts.get(2)).toBeLessThanOrEqual(610);
    });

    test('classes are separated in informative feature space', () => {
        const { X, y } = makeClassification({
            nSamples: 500,
            nFeatures: 2,
            nInformative: 2,
            nRedundant: 0,
            nClustersPerClass: 1,
            classSep: 2,
            flipY: 0,
            shuffle: false,
            randomState: 17,
        });
        const centroidOf = (label: number) => {
            const members = X.filter((_, i) => y[i] === label);
            return [mean(members.map((p) => p[0])), mean(members.map((p) => p[1]))];
        };
        const c0 = centroidOf(0);
        const c1 = centroidOf(1);
        const distance = Math.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2);
        // distinct hypercube vertices differ by 2 * classSep in at least one coordinate
        expect(distance).toBeGreaterThan(2);
    });

    test('repeated features are exact duplicates of earlier columns', () => {
        const { X } = makeClassification({
            nSamples: 50,
            nFeatures: 10,
            nInformative: 3,
            nRedundant: 2,
            nRepeated: 3,
            shuffle: false,
            randomState: 21,
        });
        for (let j = 5; j < 8; j++) {
            const matchesSomeSource = Array.from({ length: 5 }, (_, src) => src)
                .some((src) => X.every((row) => row[j] === row[src]));
            expect(matchesSomeSource).toBe(true);
        }
    });

    test('flipY introduces label noise', () => {
        const clean = makeClassification({ nSamples: 400, flipY: 0, randomState: 5, shuffle: false });
        const noisy = makeClassification({ nSamples: 400, flipY: 0.5, randomState: 5, shuffle: false });
        let flipped = 0;
        for (let i = 0; i < 400; i++) {
            if (clean.y[i] !== noisy.y[i]) {
                flipped += 1;
            }
        }
        // ~50% flip chance to a random class (may re-draw the same class)
        expect(flipped).toBeGreaterThan(50);
    });

    test('shift and scale are applied per feature', () => {
        const { X } = makeClassification({
            nSamples: 500, nFeatures: 4, nInformative: 2, nRedundant: 0,
            shift: 100, scale: 1, flipY: 0, randomState: 2,
        });
        for (let j = 0; j < 4; j++) {
            expect(columnMean(X, j)).toBeGreaterThan(90);
        }
        const scaled = makeClassification({
            nSamples: 500, nFeatures: 4, nInformative: 2, nRedundant: 0,
            shift: 100, scale: 10, flipY: 0, randomState: 2,
        });
        for (let j = 0; j < 4; j++) {
            expect(columnMean(scaled.X, j)).toBeGreaterThan(900);
        }
    });

    test('validation errors', () => {
        expect(() => makeClassification({ nFeatures: 3, nInformative: 2, nRedundant: 2 }))
            .toThrow(/nFeatures/);
        expect(() => makeClassification({ nInformative: 1, nClasses: 3, nClustersPerClass: 2 }))
            .toThrow(/nClasses \* nClustersPerClass/);
        expect(() => makeClassification({ nClasses: 3, nInformative: 3, weights: [0.5] }))
            .toThrow(/weights/);
    });
});

describe('makeRegression', () => {
    test('same seed gives identical output, different seed differs', () => {
        const a = makeRegression({ randomState: 1 });
        const b = makeRegression({ randomState: 1 });
        const c = makeRegression({ randomState: 2 });
        expect(a).toEqual(b);
        expect(a.X).not.toEqual(c.X);
    });

    test('shapes and single-target flattening', () => {
        const { X, y } = makeRegression({ nSamples: 50, nFeatures: 8, randomState: 0 });
        expect(X).toHaveLength(50);
        expect(X[0]).toHaveLength(8);
        expect(y).toHaveLength(50);
        expect(typeof (y as number[])[0]).toBe('number');
    });

    test('multi-target output shape', () => {
        const { X, y, coef } = makeRegression({
            nSamples: 30, nFeatures: 6, nTargets: 3, coef: true, randomState: 0,
        });
        expect(X).toHaveLength(30);
        expect((y as number[][])[0]).toHaveLength(3);
        expect(coef as number[][]).toHaveLength(6);
        expect((coef as number[][])[0]).toHaveLength(3);
    });

    test('y equals X . coef + bias when noise=0 (even after shuffling)', () => {
        const { X, y, coef } = makeRegression({
            nSamples: 40, nFeatures: 12, nInformative: 4, bias: 3.5, noise: 0, coef: true, randomState: 8,
        });
        const w = coef as number[];
        for (let i = 0; i < X.length; i++) {
            let expected = 3.5;
            for (let j = 0; j < w.length; j++) {
                expected += X[i][j] * w[j];
            }
            expect((y as number[])[i]).toBeCloseTo(expected, 8);
        }
    });

    test('only nInformative coefficients are non-zero', () => {
        const { coef } = makeRegression({
            nSamples: 20, nFeatures: 10, nInformative: 3, coef: true, shuffle: false, randomState: 4,
        });
        const w = coef as number[];
        for (let j = 0; j < 3; j++) {
            expect(w[j]).toBeGreaterThan(0);
            expect(w[j]).toBeLessThan(100);
        }
        for (let j = 3; j < 10; j++) {
            expect(w[j]).toBe(0);
        }
    });

    test('noise increases the residuals around the linear model', () => {
        const { X, y, coef } = makeRegression({
            nSamples: 200, nFeatures: 5, nInformative: 5, noise: 10, coef: true, randomState: 6,
        });
        const w = coef as number[];
        const residuals = X.map((row, i) => {
            let expected = 0;
            for (let j = 0; j < w.length; j++) {
                expected += row[j] * w[j];
            }
            return (y as number[])[i] - expected;
        });
        const residualStd = Math.sqrt(mean(residuals.map((r) => r * r)));
        expect(residualStd).toBeGreaterThan(7);
        expect(residualStd).toBeLessThan(13);
    });

    test('effectiveRank produces a low-rank X that still fits the linear model', () => {
        const { X, y, coef } = makeRegression({
            nSamples: 60, nFeatures: 20, nInformative: 5,
            effectiveRank: 3, tailStrength: 0.1, noise: 0, coef: true, randomState: 12,
        });
        expect(X).toHaveLength(60);
        expect(X[0]).toHaveLength(20);
        // column magnitudes shrink vs an i.i.d. gaussian design (singular values <= 1)
        const totalEnergy = mean(X.map((row) => row.reduce((acc, v) => acc + v * v, 0)));
        expect(totalEnergy).toBeLessThan(20);
        const w = coef as number[];
        for (let i = 0; i < X.length; i++) {
            let expected = 0;
            for (let j = 0; j < w.length; j++) {
                expected += X[i][j] * w[j];
            }
            expect((y as number[])[i]).toBeCloseTo(expected, 8);
        }
    });

    test('effectiveRank validation', () => {
        expect(() => makeRegression({ effectiveRank: 0 })).toThrow(/effectiveRank/);
    });
});

describe('makeMoons', () => {
    test('same seed gives identical output', () => {
        const a = makeMoons({ noise: 0.1, randomState: 5 });
        const b = makeMoons({ noise: 0.1, randomState: 5 });
        expect(a).toEqual(b);
    });

    test('shapes and label split', () => {
        const { X, y } = makeMoons({ nSamples: 101, randomState: 0 });
        expect(X).toHaveLength(101);
        expect(X[0]).toHaveLength(2);
        const counts = countLabels(y);
        expect(counts.get(0)).toBe(50);
        expect(counts.get(1)).toBe(51);
    });

    test('tuple nSamples controls per-moon counts', () => {
        const { y } = makeMoons({ nSamples: [30, 70], randomState: 0 });
        const counts = countLabels(y);
        expect(counts.get(0)).toBe(30);
        expect(counts.get(1)).toBe(70);
    });

    test('noiseless points lie on the two half circles', () => {
        const { X, y } = makeMoons({ nSamples: 200, randomState: 1 });
        for (let i = 0; i < X.length; i++) {
            const [px, py] = X[i];
            if (y[i] === 0) {
                // outer moon: unit circle around the origin, upper half
                expect(Math.hypot(px, py)).toBeCloseTo(1, 8);
                expect(py).toBeGreaterThanOrEqual(-1e-9);
            } else {
                // inner moon: unit circle around (1, 0.5), lower half
                expect(Math.hypot(px - 1, py - 0.5)).toBeCloseTo(1, 8);
                expect(py).toBeLessThanOrEqual(0.5 + 1e-9);
            }
        }
    });

    test('noise perturbs points but keeps them near the moons', () => {
        const { X, y } = makeMoons({ nSamples: 400, noise: 0.05, randomState: 2 });
        const radii = X.map(([px, py], i) => (
            y[i] === 0 ? Math.hypot(px, py) : Math.hypot(px - 1, py - 0.5)
        ));
        expect(radii.some((r) => Math.abs(r - 1) > 1e-6)).toBe(true);
        for (const r of radii) {
            expect(Math.abs(r - 1)).toBeLessThan(0.4);
        }
    });

    test('validation error for bad tuple', () => {
        expect(() => makeMoons({ nSamples: [10, 10, 10] as never })).toThrow(/two-element/);
    });
});

describe('makeCircles', () => {
    test('same seed gives identical output', () => {
        const a = makeCircles({ noise: 0.1, randomState: 5 });
        const b = makeCircles({ noise: 0.1, randomState: 5 });
        expect(a).toEqual(b);
    });

    test('noiseless radii match factor', () => {
        const { X, y } = makeCircles({ nSamples: 100, factor: 0.5, randomState: 0 });
        for (let i = 0; i < X.length; i++) {
            const r = Math.hypot(X[i][0], X[i][1]);
            expect(r).toBeCloseTo(y[i] === 0 ? 1 : 0.5, 8);
        }
    });

    test('tuple nSamples controls per-circle counts', () => {
        const { y } = makeCircles({ nSamples: [20, 80], randomState: 0 });
        const counts = countLabels(y);
        expect(counts.get(0)).toBe(20);
        expect(counts.get(1)).toBe(80);
    });

    test('factor validation', () => {
        expect(() => makeCircles({ factor: 1 })).toThrow(/factor/);
        expect(() => makeCircles({ factor: -0.1 })).toThrow(/factor/);
    });

    test('shuffle=false keeps outer circle first', () => {
        const { y } = makeCircles({ nSamples: 10, shuffle: false, randomState: 0 });
        expect(y).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    });
});
