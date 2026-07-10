import { SpectralEmbedding } from '../spectralEmbedding';

 test('basic spectral embedding', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const se = new SpectralEmbedding({ nComponents: 2, nNeighbors: 2 });
    const T = se.fitTransform(X);
    expect(T.length).toBe(4);
    expect(T[0].length).toBe(2);
});

test('spectral embedding is reproducible with randomState', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [5, 5],
        [6, 5]
    ];
    const a = new SpectralEmbedding({ nComponents: 2, nNeighbors: 2, randomState: 11 }).fitTransform(X);
    const b = new SpectralEmbedding({ nComponents: 2, nNeighbors: 2, randomState: 11 }).fitTransform(X);
    expect(b).toEqual(a);
});
