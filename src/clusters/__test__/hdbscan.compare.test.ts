import { HDBScan } from '../hdbscan';
import fs from 'fs';
import path from 'path';

// Noise (-1) is kept as-is: it must match sklearn exactly and does not
// take part in the label permutation.
function normalize(labels: number[]): number[] {
    const map = new Map<number, number>();
    let counter = 0;
    return labels.map(l => {
        if (l === -1) return -1;
        if (!map.has(l)) map.set(l, counter++);
        return map.get(l)!;
    });
}

function countClusters(labels: number[]): number {
    return new Set(labels.filter(l => l !== -1)).size;
}

const p = path.join(__dirname, '../../../test_data/hdbscan.json');
const data = JSON.parse(fs.readFileSync(p, 'utf8'));

test('compare with sklearn (min_cluster_size=8)', () => {
    const pred = new HDBScan(8).fitPredict(data.X);
    expect(countClusters(pred)).toBe(countClusters(data.expected_mcs8));
    expect(pred.filter((l: number) => l === -1).length).toBe(
        data.expected_mcs8.filter((l: number) => l === -1).length
    );
    expect(normalize(pred)).toEqual(normalize(data.expected_mcs8));
});

test('compare with sklearn (min_cluster_size=25)', () => {
    const pred = new HDBScan(25).fitPredict(data.X);
    expect(countClusters(pred)).toBe(countClusters(data.expected_mcs25));
    expect(pred.filter((l: number) => l === -1).length).toBe(
        data.expected_mcs25.filter((l: number) => l === -1).length
    );
    expect(normalize(pred)).toEqual(normalize(data.expected_mcs25));
});

test('compare with sklearn (min_cluster_size=8, cluster_selection_epsilon=2)', () => {
    const pred = new HDBScan(8, null, 2.0).fitPredict(data.X);
    expect(countClusters(pred)).toBe(countClusters(data.expected_eps2));
    expect(pred.filter((l: number) => l === -1).length).toBe(
        data.expected_eps2.filter((l: number) => l === -1).length
    );
    expect(normalize(pred)).toEqual(normalize(data.expected_eps2));
});
