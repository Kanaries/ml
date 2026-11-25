import fs from 'fs';
import path from 'path';
import { Clusters, loadModel } from '../../index';
import { EMeansTraining, nearestCenter, resolveEMeansRecipe, squaredDistance } from '../eMeansTraining';

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/e_means.json'), 'utf8'));
const { EMeans } = Clusters;

function near(actual: number[], expected: number[]) {
    expect(actual).toHaveLength(expected.length);
    actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 5));
}

test('Faiss v1.16.0 quality/throughput laws, caps and explicit batches', () => {
    for (const r of fixture.recipes) {
        const recipe = resolveEMeansRecipe(r.n, r.d, r.k, r.epochs, r.batch ?? undefined);
        expect(recipe.batchSize).toBe(r.size);
        expect(recipe.alpha0).toBeCloseTo(r.alpha, 12);
    }
});

test.each([0, 1])('Faiss CPU trace %i, including the remainder of each pass', index => {
    const t = fixture.traces[index];
    const training = new EMeansTraining(t.init.map(row => row.slice()), resolveEMeansRecipe(9, 3, 3, 2, 2),
        t.threshold, () => 0.4);
    for (const step of t.batches) {
        training.update(t.x, step.ids, 0, step.ids.length, step.closing);
        near(Array.from(training.mass), step.mass);
        training.centers.forEach((row, i) => near(row, step.centers[i]));
    }
    const centers = training.finalize();
    centers.forEach((row, i) => near(row, t.centers[i]));
    expect(t.splits).toBe(index);
    const labels = t.x.map(row => nearestCenter(row, centers));
    expect(labels).toEqual(t.labels);
    expect(t.x.reduce((s, row, i) => s + squaredDistance(row, centers[labels[i]]), 0)).toBeCloseTo(t.inertia, 5);
});

test('public training reproduces seeded CPU readout and preserves state across JSON', () => {
    const t = fixture.traces[0];
    const model = new EMeans({ nClusters: 3, epochs: 2, batchSize: 2, randomState: 17, initCenters: t.init, revivalThreshold: 0 });
    expect(model.fitPredict(t.x)).toEqual(t.labels);
    model.getCentroids().forEach((c, i) => near(c, t.centers[i]));
    expect(model.getInertia()).toBeCloseTo(t.inertia, 5);
    expect(model.predict(t.x)).toEqual(model.getLabels());
    const copy = loadModel(JSON.stringify(model)) as Clusters.EMeans;
    expect(copy.predict(t.x)).toEqual(model.getLabels());
    expect(copy.getInertia()).toBe(model.getInertia());
    expect(model.fitPredict(t.x, t.x.map(() => 1))).toEqual(t.labels);
    model.setParams({ batchSize: 1 });
    expect(model.getCentroids()).toBeNull();
});

test('exact final assignment even when last-pass assignments differ', () => {
    const X = Array.from({ length: 30 }, (_, i) => [i]);
    const model = new EMeans({ nClusters: 3, epochs: 1, initCenters: [[1], [2], [3]], batchSize: 2, randomState: 5 }).fit(X);
    const centers = model.getCentroids().map(c => c[0]);
    const distance = X.map(([x]) => centers.map(c => (x - c) ** 2));
    expect(model.getLabels()).toEqual(distance.map(row => row.indexOf(Math.min(...row))));
    expect(model.getInertia()).toBeCloseTo(distance.reduce((total, row) => total + Math.min(...row), 0), 10);
    expect(model.predict([])).toEqual([]);
});

test('single cluster closes at the sample mean; duplicates and tied assignments stay finite', () => {
    const single = new EMeans({ nClusters: 1, randomState: 2 }).fit([[1], [3], [5]]);
    expect(single.getCentroids()).toEqual([[3]]);
    expect(single.getInertia()).toBe(8);
    expect(new EMeans({ nClusters: 1 }).fitPredict([[3]])).toEqual([0]);
    const identical = new EMeans({ nClusters: 4, epochs: 6, batchSize: 1, randomState: 3 }).fit(Array(4).fill([0, 0, 0]));
    expect(identical.getLabels()).toEqual([0, 0, 0, 0]);
    expect(identical.getInertia()).toBe(0);
    const X = [[1, 2], [2, 1], [3, 1]];
    expect(new EMeans({ nClusters: 3, randomState: 4 }).fit(X).getCentroids()).toEqual(X);
    const largeBatch = new EMeans({ nClusters: 2, epochs: 5, batchSize: 100, initCenters: [[0], [100]] }).fit(Array(31).fill([1]));
    expect(largeBatch.getCentroids()).toEqual([[1], [100]]);
});

test('reject invalid params, rows, weights and incompatible prediction before changing fitted state', () => {
    for (const props of [{ epochs: 1.5 }, { batchSize: 0 }, { nClusters: 0 }, { randomState: NaN },
        { revivalThreshold: -1 }, { revivalThreshold: Infinity }, { nClusters: 2, initCenters: [[1]] }]) {
        expect(() => new EMeans(props)).toThrow();
    }
    expect(() => new EMeans().predict([])).toThrow(/fit/);
    const model = new EMeans({ nClusters: 2, randomState: 4 }).fit([[0], [10]]);
    for (const X of [[], [[1]], [[], []], [[1], [1, 2]], [[1], [NaN]], [[Infinity], [2]]]) {
        expect(() => model.fit(X)).toThrow();
    }
    for (const w of [[], [0, 1], [1, NaN]]) expect(() => model.fit([[0], [10]], w)).toThrow(/unit/);
    expect(() => model.predict([[0, 1]])).toThrow(/dimension/);
    expect(() => model.predict([[Number.MAX_VALUE]])).toThrow(/numeric range/);
    expect(model.predict([[0], [10]])).toEqual([0, 1]);
});

test('disabling revival preserves a seeded rare isolated group', () => {
    const X = Array.from({ length: 500 }, (_, i) => i % 100 === 0
        ? [20, 20, 20] : [1 + Math.sin(i * 0.93), Math.cos(i * 0.47), Math.sin(i * 0.34)]);
    const props = { nClusters: 3, randomState: 3, epochs: 5, initCenters: [X[2], X[4], X[0]] };
    const kept = new EMeans({ ...props, revivalThreshold: 0 }).fit(X);
    expect(kept.getCentroids()).toContainEqual([20, 20, 20]);
    expect(new EMeans(props).fit(X).getInertia()).toBeGreaterThan(5 * kept.getInertia());
});

test('large signed seeds yield valid repeatable shuffles', () => {
    const model = new EMeans({ nClusters: 2, randomState: -2147483646 });
    const X = [[1], [2], [20]];
    expect(model.fitPredict(X)).toEqual(model.clone().fitPredict(X));
});
