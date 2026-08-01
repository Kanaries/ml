import fs from 'fs';
import path from 'path';
import { AffinityPropagation } from '../affinityPropagation';
import { Birch } from '../birch';
import { BisectingKMeans } from '../bisectingKMeans';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));
const closeMatrix = (actual: number[][], expected: number[][], digits = 8) => actual.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expected[i][j], digits)));

test('Birch threshold CF subclusters match sklearn on separated streams', () => {
    const { X, birch } = waveC.clustering, model = new Birch({ threshold: .5, nClusters: null });
    expect(model.fitPredict(X)).toEqual(birch.labels); closeMatrix(model.subclusterCenters, birch.subcluster_centers, 12); expect(model.subclusterLabels).toEqual(birch.subcluster_labels);
    const incremental = new Birch({ threshold: .5, nClusters: null }); incremental.partialFit(X.slice(0, 4)).partialFit(X.slice(4)); expect(incremental.predict(X)).toEqual(birch.labels);
});

test('Birch uses branchingFactor CF-tree splits and sklearn agglomerative global clustering', () => {
    const fixture = waveC.clustering.birch_tree, model = new Birch({ threshold: .12, branchingFactor: 2, nClusters: 3 });
    const labels = model.fitPredict(fixture.X);
    closeMatrix(model.subclusterCenters, fixture.subcluster_centers, 10);
    const samePartition = (a: number[], b: number[]) => a.every((_, i) => a.every((__, j) => (a[i] === a[j]) === (b[i] === b[j])));
    expect(samePartition(labels, fixture.labels)).toBe(true);
    expect(samePartition(model.subclusterLabels, fixture.subcluster_labels)).toBe(true);
});

test('AffinityPropagation messages recover sklearn exemplars and labels', () => {
    const { X, affinity_propagation: fixture } = waveC.clustering, model = new AffinityPropagation({ damping: .7, preference: -20, maxIter: 500, convergenceIter: 20, randomState: 0 });
    expect(model.fitPredict(X)).toEqual(fixture.labels); expect(model.clusterCenterIndices).toEqual(fixture.center_indices); closeMatrix(model.clusterCenters, fixture.centers); expect(model.nIter).toBe(fixture.n_iter);
});

test('BisectingKMeans matches sklearn separated-cluster solution', () => {
    const { X, bisecting_kmeans: fixture } = waveC.clustering, model = new BisectingKMeans({ nClusters: 3, randomState: 0, nInit: 10 });
    expect(model.fitPredict(X)).toEqual(fixture.labels); closeMatrix(model.clusterCenters, fixture.centers, 8); expect(model.inertia).toBeCloseTo(fixture.inertia, 10);
});

test('BisectingKMeans keeps finite duplicate centers when an identical group is split', () => {
    const model = new BisectingKMeans({ nClusters: 3, randomState: 0, nInit: 1 }); model.fit([[0, 0], [0, 0], [5, 5], [5, 5]]);
    expect(model.clusterCenters).toHaveLength(3); expect(model.clusterCenters.flat().every(Number.isFinite)).toBe(true); expect(model.inertia).toBe(0);
});
