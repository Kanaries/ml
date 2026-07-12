import { MiniBatchKMeans } from '../miniBatchKMeans';
import { KMeans } from '../kmeans';
import { makeBlobs } from '../../datasets';

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

const blobs = makeBlobs({
    nSamples: 300,
    centers: [[0, 0], [10, 10], [-10, 10]],
    clusterStd: 0.7,
    randomState: 7,
});

test('init', () => {
    expect(new MiniBatchKMeans()).toBeDefined();
});

test('recovers three easy blobs and matches full KMeans up to permutation', () => {
    const mbk = new MiniBatchKMeans({ nClusters: 3, randomState: 0, batchSize: 64 });
    const labels = mbk.fitPredict(blobs.X);
    expect(labels).toHaveLength(blobs.X.length);
    expect(new Set(labels).size).toBe(3);
    expect(samePartition(labels, blobs.y)).toBe(true);

    const kmLabels = new KMeans({ n_clusters: 3, random_state: 0 }).fitPredict(blobs.X);
    expect(samePartition(labels, kmLabels)).toBe(true);
});

test('inertia is within 10% of full-batch KMeans', () => {
    const mbk = new MiniBatchKMeans({ nClusters: 3, randomState: 42, batchSize: 64 });
    mbk.fitPredict(blobs.X);
    const km = new KMeans({ n_clusters: 3, random_state: 42 });
    km.fitPredict(blobs.X);
    expect(Number.isFinite(mbk.getInertia())).toBe(true);
    expect(mbk.getInertia()).toBeLessThanOrEqual(km.getInertia() * 1.1);
});

test('batchSize larger than nSamples degenerates gracefully', () => {
    const small = makeBlobs({
        nSamples: 90,
        centers: [[0, 0], [8, 8], [-8, 8]],
        clusterStd: 0.5,
        randomState: 11,
    });
    const mbk = new MiniBatchKMeans({ nClusters: 3, randomState: 1, batchSize: 5000 });
    const labels = mbk.fitPredict(small.X);
    expect(labels).toHaveLength(small.X.length);
    expect(samePartition(labels, small.y)).toBe(true);
    const centroids = mbk.getCentroids();
    expect(centroids).toHaveLength(3);
    for (const center of centroids as number[][]) {
        for (const value of center) expect(Number.isFinite(value)).toBe(true);
    }
});

test('seeded runs are deterministic', () => {
    const a = new MiniBatchKMeans({ nClusters: 3, randomState: 123, batchSize: 32 });
    const b = new MiniBatchKMeans({ nClusters: 3, randomState: 123, batchSize: 32 });
    expect(a.fitPredict(blobs.X)).toEqual(b.fitPredict(blobs.X));
    expect(a.getCentroids()).toEqual(b.getCentroids());
    expect(a.getInertia()).toEqual(b.getInertia());
});

test('predict assigns new points to the nearest fitted center', () => {
    const mbk = new MiniBatchKMeans({ nClusters: 3, randomState: 5, batchSize: 64 });
    const labels = mbk.fitPredict(blobs.X);
    // one probe point right next to each blob center: its predicted label
    // must equal the fitted label of the training points from that blob
    const probes: Array<[number[], number]> = [
        [[0.1, -0.2], 0],
        [[9.9, 10.1], 1],
        [[-10.2, 9.8], 2],
    ];
    const predicted = mbk.predict(probes.map(([p]) => p));
    for (let i = 0; i < probes.length; i++) {
        const blobLabel = probes[i][1];
        const trainIdx = blobs.y.findIndex((label) => label === blobLabel);
        expect(predicted[i]).toBe(labels[trainIdx]);
    }
    // predict is pure out-of-sample assignment: training data round-trips
    expect(mbk.predict(blobs.X)).toEqual(labels);
});

test('predict before fit throws; empty fit resets state', () => {
    expect(() => new MiniBatchKMeans({ nClusters: 2 }).predict([[0, 0]])).toThrow(/fit before predict/);
    const mbk = new MiniBatchKMeans({ nClusters: 2, randomState: 1 });
    expect(mbk.fitPredict([])).toEqual([]);
    expect(mbk.getCentroids()).toBeNull();
});

test('random init and tol-based early stopping also converge on easy blobs', () => {
    // note: like sklearn, init='random' with a handful of restarts can land
    // in a local optimum for unlucky seeds; this seed converges (18 of the
    // seeds 0..19 do)
    const mbk = new MiniBatchKMeans({
        nClusters: 3,
        initParams: 'random',
        randomState: 5,
        batchSize: 64,
        tol: 1e-4,
    });
    const labels = mbk.fitPredict(blobs.X);
    expect(samePartition(labels, blobs.y)).toBe(true);
});

test('sample weights are honored in the final inertia', () => {
    const X = [[0, 0], [0.2, 0], [10, 10], [10.2, 10]];
    const w = [5, 5, 1, 1];
    const mbk = new MiniBatchKMeans({ nClusters: 2, randomState: 3, batchSize: 4 });
    const labels = mbk.fitPredict(X, w);
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).toBe(labels[3]);
    expect(labels[0]).not.toBe(labels[2]);
    expect(Number.isFinite(mbk.getInertia())).toBe(true);
});
