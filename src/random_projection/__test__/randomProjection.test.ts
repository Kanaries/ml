import fs from 'fs';
import path from 'path';
import { GaussianRandomProjection, SparseRandomProjection } from '../randomProjection';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

test('auto random projection dimension matches sklearn Johnson-Lindenstrauss bound', () => {
    const X = Array.from({ length: 100 }, (_, i) => Array.from({ length: 300 }, (_, j) => Math.sin(i + j)));
    const model = new GaussianRandomProjection({ nComponents: 'auto', eps: .5, randomState: 0 }); model.fit(X);
    expect(model.components).toHaveLength(waveC.random_projection.jl_min_dim);
    expect(model.transform(X)[0]).toHaveLength(waveC.random_projection.jl_min_dim);
});

test('auto random projection rejects the zero-dimensional single-sample bound', () => {
    expect(() => new GaussianRandomProjection({ nComponents: 'auto' }).fit([[1, 2]])).toThrow(/invalid target dimension of 0/);
});

test('Gaussian and sparse projections are deterministic and support pseudo-inverse reconstruction', () => {
    const X = Array.from({ length: 12 }, (_, i) => Array.from({ length: 8 }, (_, j) => Math.sin(i * 3 + j)));
    for (const create of [() => new GaussianRandomProjection({ nComponents: 5, randomState: 4, computeInverseComponents: true }), () => new SparseRandomProjection({ nComponents: 5, randomState: 4, computeInverseComponents: true })]) {
        const model = create(), replay = create(); model.fit(X); replay.fit(X); expect(model.components).toEqual(replay.components);
        const projected = model.transform(X); expect(projected).toHaveLength(X.length); expect(model.inverseTransform(projected).flat().every(Number.isFinite)).toBe(true);
    }
});

test('Gaussian and sparse projections preserve pairwise geometry on the sklearn fixture', () => {
    const fixture = waveC.random_projection, X = fixture.geometry_X as number[][];
    const distance = (a: number[], b: number[]) => Math.sqrt(a.reduce((sum, value, j) => sum + (value - b[j]) ** 2, 0));
    for (const model of [new GaussianRandomProjection({ nComponents: fixture.geometry_components, randomState: 17 }), new SparseRandomProjection({ nComponents: fixture.geometry_components, randomState: 17 })]) {
        const projected = model.fitTransform(X);
        const ratios: number[] = [];
        for (let i = 0; i < X.length; i++) for (let j = i + 1; j < X.length; j++) ratios.push(distance(projected[i], projected[j]) / distance(X[i], X[j]));
        const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
        expect(mean).toBeGreaterThan(.75); expect(mean).toBeLessThan(1.25);
        expect(ratios.filter(value => value >= .5 && value <= 1.5).length / ratios.length).toBeGreaterThan(.95);
    }
});
