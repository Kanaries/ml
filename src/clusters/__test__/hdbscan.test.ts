import { HDBScan } from '../hdbscan';
import fs from 'fs';
import path from 'path';

interface IFixture {
    X: number[][];
    expected_mcs8: number[];
    expected_mcs25: number[];
    expected_eps2: number[];
}

function loadFixture(): IFixture {
    const p = path.join(__dirname, '../../../test_data/hdbscan.json');
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function countClusters(labels: number[]): number {
    return new Set(labels.filter(l => l !== -1)).size;
}

function makeCircles(n1: number, n2: number, r1: number, r2: number): number[][] {
    const data: number[][] = [];
    for (let i = 0; i < n1; i++) {
        const angle = (2 * Math.PI * i) / n1;
        data.push([r1 * Math.cos(angle), r1 * Math.sin(angle)]);
    }
    for (let j = 0; j < n2; j++) {
        const angle = (2 * Math.PI * j) / n2;
        data.push([r2 * Math.cos(angle), r2 * Math.sin(angle)]);
    }
    return data;
}

function reAssignLabel(labels: number[]): number[] {
    const encoder: Map<number, number> = new Map();
    let ans: number[] = [];
    let counter = 0;
    for (let i = 0; i < labels.length; i++) {
        if (!encoder.has(labels[i])) {
            encoder.set(labels[i], counter++);
        }
        ans.push(encoder.get(labels[i])!);
    }
    return ans;
}

test('init', () => {
    const hdbscan = new HDBScan();
    expect(hdbscan).toBeDefined();
});

test('varying density: finds all three clusters plus noise', () => {
    // Two tight clusters (std 0.3) and one sparse cluster (std 2.0): a
    // fixed-eps DBSCAN cannot recover all three at once, HDBSCAN must.
    const { X } = loadFixture();
    const hdbscan = new HDBScan(8);
    const labels = hdbscan.fitPredict(X);
    expect(countClusters(labels)).toBe(3);
    expect(labels.filter(l => l === -1).length).toBeGreaterThan(0);
});

test('min_cluster_size changes the clustering', () => {
    const { X } = loadFixture();
    const labels8 = new HDBScan(8).fitPredict(X);
    const labels25 = new HDBScan(25).fitPredict(X);
    expect(labels8).not.toEqual(labels25);
});

test('cluster_selection_epsilon merges fine-grained clusters', () => {
    const { X } = loadFixture();
    const labelsNoEps = new HDBScan(8).fitPredict(X);
    const labelsHugeEps = new HDBScan(8, null, 1000).fitPredict(X);
    expect(countClusters(labelsHugeEps)).toBeGreaterThanOrEqual(1);
    expect(countClusters(labelsHugeEps)).toBeLessThan(countClusters(labelsNoEps));
});

test('hdbscan two circles', () => {
    // sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3) labels the
    // inner ring 0 and the outer ring 1 with no noise.
    const X = makeCircles(20, 20, 1, 5);
    const hdbscan = new HDBScan(3, 3);
    const rawLabels = hdbscan.fitPredict(X);
    expect(rawLabels.filter(l => l === -1).length).toBe(0);
    const labels = reAssignLabel(rawLabels);
    const expected: number[] = [];
    for (let i = 0; i < 20; i++) expected.push(0);
    for (let i = 0; i < 20; i++) expected.push(1);
    expect(labels).toEqual(expected);
});

test('identical points become noise', () => {
    // matches sklearn: no density gradient exists, so nothing is a cluster
    const X: number[][] = [];
    for (let i = 0; i < 10; i++) X.push([1, 1]);
    const labels = new HDBScan(5).fitPredict(X);
    expect(labels).toEqual(new Array(10).fill(-1));
});

test('tiny inputs do not crash', () => {
    expect(new HDBScan().fitPredict([])).toEqual([]);
    expect(new HDBScan().fitPredict([[0, 0]])).toEqual([-1]);
    expect(new HDBScan().fitPredict([[0, 0], [1, 1]])).toEqual([-1, -1]);
});

test('getLabels and getProbabilities', () => {
    const { X } = loadFixture();
    const hdbscan = new HDBScan(8);
    const labels = hdbscan.fitPredict(X);
    expect(hdbscan.getLabels()).toEqual(labels);
    const probs = hdbscan.getProbabilities();
    expect(probs.length).toBe(X.length);
    for (let i = 0; i < probs.length; i++) {
        expect(probs[i]).toBeGreaterThanOrEqual(0);
        expect(probs[i]).toBeLessThanOrEqual(1);
        if (labels[i] === -1) {
            expect(probs[i]).toBe(0);
        } else {
            expect(probs[i]).toBeGreaterThan(0);
        }
    }
});
