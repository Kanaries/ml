import { BisectingKMeans } from '../bisectingKMeans';
import { loadModel } from '../../base';

function overlappingCloud(): number[][] {
    let state = 12;
    const random = () => (state = state * 16807 % 2147483647) / 2147483647;
    return Array.from({ length: 50 }, () => [random() * 10, random() * 10]);
}

test.each(['biggestInertia', 'largestCluster'] as const)('%s preserves training assignments along the tree', bisectingStrategy => {
    const X = overlappingCloud();
    const model = new BisectingKMeans({ nClusters: 4, randomState: 42, bisectingStrategy });
    const labels = model.fitPredict(X);
    expect(model.predict(X)).toEqual(labels);
    const centers = model.clusterCenters;
    const inertia = X.reduce((sum, row, i) => sum + row.reduce((s, x, j) => s + (x - centers[labels[i]][j]) ** 2, 0), 0);
    expect(model.inertia).toBeCloseTo(inertia, 10);
});

test('new points follow the parent boundary even when another leaf center is closer', () => {
    const X = [[-10], [-9], [0], [1], [2], [20]];
    const model = new BisectingKMeans({ nClusters: 3, nInit: 10, randomState: 0 });
    expect(model.fitPredict(X)).toEqual([0, 0, 1, 1, 1, 2]);
    // Root split centers are -3.2 and 20: boundary 8.4. The final leaf
    // centers 1 and 20 have boundary 10.5, which must not replace the root.
    expect(model.predict([[8], [9], [10]])).toEqual([1, 2, 2]);
    const restored = loadModel(JSON.parse(JSON.stringify(model.toJSON()))) as BisectingKMeans;
    expect(restored.predict([[8], [9], [10]])).toEqual([1, 2, 2]);
    expect(restored.predict(X)).toEqual(model.predict(X));
});

test('early stopping keeps assignment centers distinct from updated leaf means', () => {
    const X = overlappingCloud();
    const model = new BisectingKMeans({ nClusters: 4, maxIter: 1, randomState: 42 });
    expect(model.fitPredict(X)).toEqual(model.predict(X));
});

test('single-cluster and duplicate-center trees keep valid labels', () => {
    for (const nClusters of [1, 3]) {
        const X = [[0], [0], [5], [5]];
        const model = new BisectingKMeans({ nClusters, randomState: 0 });
        const labels = model.fitPredict(X);
        expect(model.predict(X)).toEqual(labels);
        expect(model.clusterCenters).toHaveLength(nClusters);
        expect(model.clusterCenters.flat().every(Number.isFinite)).toBe(true);
    }
});

test('refitting replaces the prediction tree', () => {
    const model = new BisectingKMeans({ nClusters: 3, randomState: 0 });
    model.fit(overlappingCloud());
    const X = [[100], [101], [110], [111], [120], [121]];
    const labels = model.fitPredict(X);
    const fresh = new BisectingKMeans({ nClusters: 3, randomState: 0 });
    expect(labels).toEqual(fresh.fitPredict(X));
    expect(model.predict(X)).toEqual(labels);
});

test('legacy models without a saved hierarchy retain their previous nearest-leaf behavior', () => {
    const model = new BisectingKMeans({ nClusters: 3, nInit: 10, randomState: 0 });
    model.fit([[-10], [-9], [0], [1], [2], [20]]);
    const legacy = JSON.parse(JSON.stringify(model.toJSON()));
    delete legacy.state.treeState;
    const restored = loadModel(legacy) as BisectingKMeans;
    expect(restored.predict([[9]])).toEqual([1]);
    restored.fit([[-10], [-9], [0], [1], [2], [20]]);
    expect(restored.predict([[9]])).toEqual([2]);
});
