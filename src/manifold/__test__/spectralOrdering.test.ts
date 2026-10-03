import { Isomap } from '../isomap';
import { SpectralEmbedding } from '../spectralEmbedding';
import { loadModel } from '../../base';

const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);

test('Isomap retains positive modes ahead of larger-magnitude negative eigenvalues', () => {
    const X = Array.from({ length: 12 }, (_, i) => [Math.cos(i * Math.PI / 6), Math.sin(i * Math.PI / 6)]);
    const model = new Isomap({ nNeighbors: 2, nComponents: 3 });
    const Y = model.fitTransform(X);
    // sklearn 1.7.2 / NumPy: eigenvalues of the centered cycle geodesic kernel.
    const expected = [12, 12, 1.6076951545867373];
    const columns = expected.map((_, j) => Y.map(row => row[j]));
    columns.forEach((column, j) => expect(dot(column, column)).toBeCloseTo(expected[j], 8));
    for (let i = 0; i < columns.length; i++) for (let j = 0; j < i; j++) {
        expect(dot(columns[i], columns[j])).toBeCloseTo(0, 10);
    }
    const transformed = model.transform(X);
    transformed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(Y[i][j], 8)));
    const restored = loadModel(JSON.parse(JSON.stringify(model.toJSON()))) as Isomap;
    expect(restored.transform([[.7, .7]])).toEqual(model.transform([[.7, .7]]));
});

test('SpectralEmbedding selects low-frequency modes of its graph, not negative dominant modes', () => {
    const model = new SpectralEmbedding({ nNeighbors: 2, nComponents: 2, randomState: 1 });
    const Y = model.fitTransform([[0], [1], [2], [3], [4], [5]]);
    // Symmetric two-neighbor graph, including the implementation's self loops.
    const W = [
        [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1],
    ];
    const degrees = W.map(row => row.reduce((a, b) => a + b, 0));
    const A = W.map((row, i) => row.map((value, j) => value / Math.sqrt(degrees[i] * degrees[j])));
    // NumPy eigh spectrum: [1, .8603796100, 1/6, 0, 0, -.1937129434].
    const eigenvalues = [.8603796100280632, 1 / 6];
    const vectors = eigenvalues.map((_, j) => Y.map((row, i) => row[j] * Math.sqrt(degrees[i])));
    vectors.forEach((v, j) => {
        const Av = A.map(row => dot(row, v));
        expect(dot(v, v)).toBeCloseTo(1, 10);
        expect(dot(v, Av)).toBeCloseTo(eigenvalues[j], 10);
        expect(dot(v, degrees.map(Math.sqrt))).toBeCloseTo(0, 10);
        Av.forEach((value, i) => expect(value).toBeCloseTo(eigenvalues[j] * v[i], 10));
    });
    expect(dot(vectors[0], vectors[1])).toBeCloseTo(0, 10);
});
