import { KMeans } from '../kmeans';

test('init', () => {
    const kmeans = new KMeans();
    expect(kmeans).toBeDefined();
})

test('kmeans results', () => {
    const X = [
        [0, 0],
        [0.5, 0],
        [0.5, 1],
        [1, 1],
    ];
    const sampleWeights = [3, 1, 1, 3];
    const initCenters = [[0, 0], [1, 1]];
    const expected = [0, 0, 1, 1];

    const kmeans = new KMeans(2, 0.05, initCenters);
    const result = kmeans.fitPredict(X, sampleWeights);

    expect(result).toEqual(expected);
})
function reAssignLabel (labels: number[]): number[] {
    const encoder: Map<number, number> = new Map();
    let ans: number[] = [];
    let counter = 0;
    for (let i = 0; i < labels.length; i++) {
        if (!encoder.has(labels[i])) {
            encoder.set(labels[i], counter++);
        }
        ans.push(encoder.get(labels[i]))
    }
    return ans;
}
test('kmeans bad centers', () => {
    const X = [
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ];
    const initCenters = [
        [+0, 1, 0, 0],
        [0.2, 0, 0.2, 0.2],
        [+0, 0, 0, 0],
    ];
    const expected = [0, 1, 2, 1, 1, 2];

    const kmeans = new KMeans(3, 0.05, initCenters, 10);
    const result = reAssignLabel(kmeans.fitPredict(X));

    expect(result).toEqual(expected);
});

test('kmeans empty cluster is relocated without NaN', () => {
    const X = [
        [0, 0],
        [0.1, 0],
        [10, 10],
        [10.1, 10],
    ];
    // The third center is far from every sample, so it gets no members on
    // the first assignment and used to collapse to [NaN, NaN] (0/0).
    const initCenters = [
        [0, 0],
        [0.05, 0],
        [100, 100],
    ];
    const kmeans = new KMeans(3, 1e-4, initCenters);
    const labels = kmeans.fitPredict(X);
    const centroids = kmeans.getCentroids();

    expect(centroids).toHaveLength(3);
    for (const center of centroids) {
        for (const value of center) {
            expect(Number.isFinite(value)).toBe(true);
        }
    }
    expect(labels.every(l => l >= 0 && l < 3)).toBe(true);
    expect(Number.isFinite(kmeans.getInertia())).toBe(true);
});

test('kmeans audit counterexample keeps finite centroids and exposes inertia', () => {
    const kmeans = new KMeans(3, 1e-4, undefined, 30, 42);
    kmeans.fitPredict([
        [0, 0],
        [0.1, 0],
        [10, 10],
        [10.1, 10],
    ]);
    const centroids = kmeans.getCentroids();
    expect(centroids).toHaveLength(3);
    for (const center of centroids) {
        for (const value of center) {
            expect(Number.isFinite(value)).toBe(true);
        }
    }
    expect(Number.isFinite(kmeans.getInertia())).toBe(true);
});

test('kmeans refit on new data matches a fresh fit', () => {
    const A = [
        [0, 0],
        [1, 1],
    ];
    const B = [
        [10, 10],
        [10.1, 10],
        [20, 20],
        [20.1, 20],
    ];
    const km = new KMeans(2, 1e-4, undefined, 30, 7);
    km.fitPredict(A);
    const second = km.fitPredict(B);
    const fresh = new KMeans(2, 1e-4, undefined, 30, 7).fitPredict(B);

    expect(reAssignLabel(second)).toEqual(reAssignLabel(fresh));
    for (const center of km.getCentroids()) {
        for (const value of center) {
            expect(Number.isFinite(value)).toBe(true);
        }
    }
});

test('kmeans is reproducible with the same randomState', () => {
    const X = [
        [0, 0], [0.2, 0.1], [-0.1, 0.3],
        [5, 5], [5.2, 5.1], [4.9, 5.3],
        [10, -10], [10.2, -9.9], [9.8, -10.1],
    ];
    const kmA = new KMeans(3, 1e-4, undefined, 30, 123);
    const kmB = new KMeans(3, 1e-4, undefined, 30, 123);
    const labelsA = kmA.fitPredict(X);
    const labelsB = kmB.fitPredict(X);

    expect(labelsA).toEqual(labelsB);
    expect(kmA.getCentroids()).toEqual(kmB.getCentroids());
    expect(kmA.getInertia()).toEqual(kmB.getInertia());
});