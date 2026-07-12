import { AgglomerativeClustering, AgglomerativeLinkage } from '../agglomerativeClustering';
import { makeBlobs } from '../../datasets';

const LINKAGES: AgglomerativeLinkage[] = ['ward', 'complete', 'average', 'single'];

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

test('init', () => {
    expect(new AgglomerativeClustering()).toBeDefined();
});

describe('three well-separated blobs are recovered by every linkage', () => {
    const { X, y } = makeBlobs({
        nSamples: 60,
        centers: [[0, 0], [10, 10], [-10, 10]],
        clusterStd: 0.5,
        randomState: 42,
    });
    for (const linkage of LINKAGES) {
        it(`linkage=${linkage}`, () => {
            const model = new AgglomerativeClustering({ nClusters: 3, linkage });
            const labels = model.fitPredict(X);
            expect(labels).toHaveLength(X.length);
            expect(new Set(labels).size).toBe(3);
            expect(samePartition(labels, y)).toBe(true);
        });
    }
});

describe('hand-verified tiny 1-D case', () => {
    // points 0, 1, 10, 11: the only sensible 2-clustering is {0,1} | {10,11}
    const X = [[0], [1], [10], [11]];
    for (const linkage of LINKAGES) {
        it(`nClusters=2, linkage=${linkage} -> {0,1} | {10,11}`, () => {
            const labels = new AgglomerativeClustering({ nClusters: 2, linkage }).fitPredict(X);
            expect(labels[0]).toBe(labels[1]);
            expect(labels[2]).toBe(labels[3]);
            expect(labels[0]).not.toBe(labels[2]);
        });
    }
});

test('children/distances have sklearn shapes and sorted distances', () => {
    const X = [[0], [1], [10], [11], [12]];
    const model = new AgglomerativeClustering({ nClusters: 2, linkage: 'average' });
    model.fitPredict(X);
    const children = model.getChildren();
    const distances = model.getDistances();
    expect(children).toHaveLength(X.length - 1);
    for (const pair of children) {
        expect(pair).toHaveLength(2);
        // node ids are leaves (< n) or merge nodes (n + row), all < 2n - 1
        for (const node of pair) {
            expect(node).toBeGreaterThanOrEqual(0);
            expect(node).toBeLessThan(2 * X.length - 1);
        }
    }
    expect(distances).toHaveLength(X.length - 1);
    for (let i = 1; i < distances.length; i++) {
        expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1]);
    }
    // every merge node referenced by a later row must exist
    const seen = new Set<number>();
    children.forEach((pair, row) => {
        for (const node of pair) {
            if (node >= X.length) expect(seen.has(node)).toBe(true);
        }
        seen.add(X.length + row);
    });
});

test('ward with a non-euclidean metric throws', () => {
    expect(() => new AgglomerativeClustering({ linkage: 'ward', metric: 'manhattan' }))
        .toThrow(/euclidean/);
    // non-ward linkages accept other metrics
    expect(() => new AgglomerativeClustering({ linkage: 'average', metric: 'manhattan' }))
        .not.toThrow();
});

test('exactly one of nClusters / distanceThreshold must be set', () => {
    expect(() => new AgglomerativeClustering({ nClusters: 2, distanceThreshold: 1 }))
        .toThrow(/Exactly one/);
    expect(() => new AgglomerativeClustering({ nClusters: null, distanceThreshold: null }))
        .toThrow(/Exactly one/);
    expect(() => new AgglomerativeClustering({ nClusters: null, distanceThreshold: 1 }))
        .not.toThrow();
});

describe('distanceThreshold cutting on the tiny case', () => {
    // X = [0], [1], [10], [11]. Every linkage first merges {0,1} and
    // {10,11}, both at distance 1. The final cross merge happens at:
    //   single:   min distance          = 9
    //   average:  mean of {9,10,10,11}  = 10
    //   complete: max distance          = 11
    //   ward:     sqrt(2*|A||B|/(|A|+|B|)) * |centroid gap|
    //             = sqrt(2*2*2/4) * |0.5 - 10.5| = sqrt(2)*10 ~= 14.14
    // So a threshold of 5 refuses only the cross merge (2 clusters) and a
    // threshold of 0.5 refuses all three merges (4 clusters), for every
    // linkage.
    const X = [[0], [1], [10], [11]];
    for (const linkage of LINKAGES) {
        it(`linkage=${linkage}: threshold 5 -> 2 clusters, threshold 0.5 -> 4`, () => {
            const two = new AgglomerativeClustering({ nClusters: null, distanceThreshold: 5, linkage });
            const labels = two.fitPredict(X);
            expect(two.getNClusters()).toBe(2);
            expect(new Set(labels).size).toBe(2);
            expect(labels[0]).toBe(labels[1]);
            expect(labels[2]).toBe(labels[3]);

            const four = new AgglomerativeClustering({ nClusters: null, distanceThreshold: 0.5, linkage });
            expect(new Set(four.fitPredict(X)).size).toBe(4);
            expect(four.getNClusters()).toBe(4);
        });
    }
    it('exact cross-merge distances match the hand computation', () => {
        const expected: Record<AgglomerativeLinkage, number> = {
            single: 9,
            average: 10,
            complete: 11,
            ward: Math.sqrt(2) * 10,
        };
        for (const linkage of LINKAGES) {
            const model = new AgglomerativeClustering({ nClusters: 2, linkage });
            model.fitPredict(X);
            const distances = model.getDistances();
            expect(distances[0]).toBeCloseTo(1, 10);
            expect(distances[1]).toBeCloseTo(1, 10);
            expect(distances[2]).toBeCloseTo(expected[linkage], 10);
        }
    });
});

test('degenerate inputs: empty, single sample, nClusters > nSamples', () => {
    const model = new AgglomerativeClustering({ nClusters: 2 });
    expect(model.fitPredict([])).toEqual([]);
    expect(new AgglomerativeClustering({ nClusters: 1 }).fitPredict([[1, 2]])).toEqual([0]);
    expect(() => model.fitPredict([[1, 2]])).toThrow(/Cannot extract/);
});

test('refit is deterministic and matches a fresh instance', () => {
    const { X } = makeBlobs({ nSamples: 40, centers: [[0, 0], [8, 8]], clusterStd: 1, randomState: 3 });
    const a = new AgglomerativeClustering({ nClusters: 2, linkage: 'complete' });
    const first = a.fitPredict(X).slice();
    const second = a.fitPredict(X);
    expect(second).toEqual(first);
    expect(new AgglomerativeClustering({ nClusters: 2, linkage: 'complete' }).fitPredict(X)).toEqual(first);
});
