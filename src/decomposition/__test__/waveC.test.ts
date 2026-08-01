import fs from 'fs';
import path from 'path';
import { FactorAnalysis } from '../factorAnalysis';
import { LatentDirichletAllocation } from '../latentDirichletAllocation';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

test('FactorAnalysis EM matches sklearn covariance and latent geometry', () => {
    const fixture = waveC.decomposition.factor_analysis, model = new FactorAnalysis({ nComponents: 2, tol: 1e-4, maxIter: 1000 }); model.fit(fixture.X);
    model.mean.forEach((value, i) => expect(value).toBeCloseTo(fixture.mean[i], 12));
    model.noiseVariance.forEach((value, i) => expect(value).toBeCloseTo(fixture.noise_variance[i], 5));
    model.getCovariance().forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.covariance[i][j], 5)));
    expect(model.nIter).toBe(fixture.n_iter);
    const transformed = model.transform(fixture.rows.map((i: number) => fixture.X[i]));
    transformed.forEach((row, i) => row.forEach((value, j) => expect(Math.abs(value)).toBeCloseTo(Math.abs(fixture.transform[i][j]), 4)));
    const rotated = new FactorAnalysis({ nComponents: 2, rotation: 'varimax', tol: 1e-4, maxIter: 1000 }); rotated.fit(fixture.X);
    rotated.components.forEach((row, i) => row.forEach((value, j) => expect(Math.abs(value)).toBeCloseTo(Math.abs(fixture.varimax_components[i][j]), 4)));
    rotated.transform(fixture.rows.map((i: number) => fixture.X[i])).forEach((row, i) => row.forEach((value, j) => expect(Math.abs(value)).toBeCloseTo(Math.abs(fixture.varimax_transform[i][j]), 4)));
});

test('FactorAnalysis with zero components keeps sklearn output shapes', () => {
    const X = [[0, 1, 2], [1, 2, 4], [2, 4, 8]], model = new FactorAnalysis({ nComponents: 0 }); model.fit(X);
    expect(model.transform(X)).toEqual([[], [], []]);
    expect(model.getCovariance()).toHaveLength(3); expect(model.getCovariance().every(row => row.length === 3)).toBe(true);
    const rotated = new FactorAnalysis({ nComponents: 0, rotation: 'varimax' }); expect(() => rotated.fit(X)).not.toThrow(); expect(rotated.transform(X)).toEqual([[], [], []]);
});

test('LatentDirichletAllocation separates the same two word topics as sklearn', () => {
    const fixture = waveC.decomposition.lda, model = new LatentDirichletAllocation({ nComponents: 2, maxIter: 30, randomState: 0 }); const transformed = model.fitTransform(fixture.X);
    const topics = model.components.map(row => { const sum = row.reduce((a, b) => a + b, 0); return row.map(value => value / sum); });
    const similarity = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0) / Math.sqrt(a.reduce((s, v) => s + v * v, 0) * b.reduce((s, v) => s + v * v, 0));
    const direct = [similarity(topics[0], fixture.topics[0]), similarity(topics[1], fixture.topics[1])], swapped = [similarity(topics[0], fixture.topics[1]), similarity(topics[1], fixture.topics[0])];
    const aligned = direct[0] + direct[1] >= swapped[0] + swapped[1] ? direct : swapped; aligned.forEach(value => expect(value).toBeGreaterThan(.98));
    transformed.forEach(row => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12));
    const swap = direct[0] + direct[1] < swapped[0] + swapped[1];
    transformed.forEach((row, i) => row.forEach((value, topic) => expect(value).toBeCloseTo(fixture.transform[i][swap ? 1 - topic : topic], 1)));
    expect(() => model.transform(fixture.X.map((row: number[]) => row.slice(1)))).toThrow(/feature count/);
});
