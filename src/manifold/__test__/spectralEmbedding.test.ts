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
