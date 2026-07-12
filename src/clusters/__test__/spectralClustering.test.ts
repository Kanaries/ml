import { SpectralClustering, jacobiEigenSymmetric } from '../spectralClustering';
import { KMeans } from '../kmeans';
import { makeCircles, makeBlobs } from '../../datasets';

/** true iff two labelings induce the same partition (same co-membership). */
function samePartition(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        for (let j = i + 1; j < a.length; j++) {
            if ((a[i] === a[j]) !== (b[i] === b[j])) return false;
        }
    }
    return true;
}

/** binary-labeling accuracy up to label permutation */
function binaryAccuracy(pred: number[], truth: number[]): number {
    let agree = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === truth[i]) agree++;
    }
    return Math.max(agree, pred.length - agree) / pred.length;
}

test('init', () => {
    expect(new SpectralClustering()).toBeDefined();
});

test('jacobi eigensolver recovers a known spectrum', () => {
    // symmetric matrix with eigenvalues 1, 2, 4 (built as Q diag Q^T)
    const A = [
        [2.5, -1.5, 0],
        [-1.5, 2.5, 0],
        [0, 0, 2],
    ];
    const { values, vectors } = jacobiEigenSymmetric(A);
    expect(values[0]).toBeCloseTo(1, 8);
    expect(values[1]).toBeCloseTo(2, 8);
    expect(values[2]).toBeCloseTo(4, 8);
    // eigenpair residual ||Av - lambda v|| ~= 0 and unit-norm vectors
    for (let e = 0; e < 3; e++) {
        const v = vectors[e];
        let norm = 0;
        for (let i = 0; i < 3; i++) norm += v[i] * v[i];
        expect(Math.sqrt(norm)).toBeCloseTo(1, 8);
        for (let i = 0; i < 3; i++) {
            const av = A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2];
            expect(av).toBeCloseTo(values[e] * v[i], 8);
        }
    }
});

test('rbf affinity separates two concentric circles where raw KMeans cannot', () => {
    const { X, y } = makeCircles({ nSamples: 160, factor: 0.5, noise: 0.02, randomState: 42 });

    const spectral = new SpectralClustering({
        nClusters: 2,
        affinity: 'rbf',
        gamma: 50,
        randomState: 0,
        nInit: 10,
    });
    const spectralLabels = spectral.fitPredict(X);
    expect(binaryAccuracy(spectralLabels, y)).toBeGreaterThanOrEqual(0.95);

    // KMeans on the raw coordinates splits the plane by a line through the
    // shared center, so it cannot do much better than chance on circles
    const kmeansLabels = new KMeans({ n_clusters: 2, random_state: 0 }).fitPredict(X);
    expect(binaryAccuracy(kmeansLabels, y)).toBeLessThan(0.75);
});

test('nearestNeighbors affinity recovers three well-separated blobs', () => {
    const { X, y } = makeBlobs({
        nSamples: 90,
        centers: [[0, 0], [10, 10], [-10, 10]],
        clusterStd: 0.6,
        randomState: 7,
    });
    const model = new SpectralClustering({
        nClusters: 3,
        affinity: 'nearestNeighbors',
        nNeighbors: 10,
        randomState: 1,
    });
    const labels = model.fitPredict(X);
    expect(new Set(labels).size).toBe(3);
    expect(samePartition(labels, y)).toBe(true);
});

test('seeded runs are deterministic', () => {
    const { X } = makeCircles({ nSamples: 80, factor: 0.5, noise: 0.03, randomState: 5 });
    const props = { nClusters: 2, affinity: 'rbf' as const, gamma: 40, randomState: 123 };
    const a = new SpectralClustering(props).fitPredict(X);
    const b = new SpectralClustering(props).fitPredict(X);
    expect(b).toEqual(a);
    // getLabels mirrors the last fit
    const model = new SpectralClustering(props);
    expect(model.fitPredict(X)).toEqual(model.getLabels());
});

test('validates affinity, assignLabels and cluster count', () => {
    expect(() => new SpectralClustering({ affinity: 'precomputed' as never })).toThrow(/affinity/);
    expect(() => new SpectralClustering({ assignLabels: 'discretize' as never })).toThrow(/kmeans/);
    expect(() => new SpectralClustering({ nClusters: 5 }).fitPredict([[0, 0], [1, 1]]))
        .toThrow(/Cannot extract/);
    expect(new SpectralClustering({ nClusters: 2 }).fitPredict([])).toEqual([]);
});
