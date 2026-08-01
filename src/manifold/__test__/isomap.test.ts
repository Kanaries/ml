import { Isomap } from '../isomap';

test('Isomap uses shortest-path geodesics and supports out-of-sample transform', () => {
    const X = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];
    const model = new Isomap({ nNeighbors: 1, nComponents: 1 });
    const embedding = model.fitTransform(X);
    expect(model.distMatrix[0][4]).toBeCloseTo(4, 12);
    expect(embedding).toHaveLength(5);
    embedding.forEach(row => expect(row).toHaveLength(1));
    for (let i = 0; i < X.length; i++) for (let j = 0; j < X.length; j++) {
        expect(Math.abs(embedding[i][0] - embedding[j][0])).toBeCloseTo(Math.abs(i - j), 6);
    }
    const transformed = model.transform([[2.5, 0]]);
    expect(transformed).toHaveLength(1);
    expect(transformed[0]).toHaveLength(1);
    expect(Number.isFinite(transformed[0][0])).toBe(true);
});

test('Isomap preserves the requested output width for zero-eigenvalue components', () => {
    const model = new Isomap({ nNeighbors: 2, nComponents: 3 });
    const embedding = model.fitTransform([[0, 0], [1, 0], [2, 0], [3, 0]]);
    expect(embedding).toHaveLength(4);
    embedding.forEach(row => {
        expect(row).toHaveLength(3);
        expect(row.every(Number.isFinite)).toBe(true);
    });
    expect(model.transform([[1.5, 0]])[0]).toHaveLength(3);
});

test('Isomap bridges disconnected neighbor components like sklearn', () => {
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    const model = new Isomap({ nNeighbors: 1, nComponents: 2 });
    const embedding = model.fitTransform([[0], [1], [10], [11]]);
    expect(embedding).toHaveLength(4);
    embedding.forEach(row => expect(row).toHaveLength(2));
    expect(model.distMatrix[0][3]).toBeCloseTo(11, 12);
    expect(warning).toHaveBeenCalledWith(expect.stringContaining('2 connected components'));
    warning.mockRestore();
});
