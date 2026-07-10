import { kmeansPlusPlus } from '../kmeans_plusplus';

test('kmeans++ basic', () => {
    const X = [
        [0, 0],
        [10, 10],
        [5, 0],
        [0, 5]
    ];
    const originalRandom = Math.random;
    const mock = jest
        .spyOn(Math, 'random')
        .mockReturnValueOnce(0.1) // first center -> index 0
        .mockReturnValueOnce(0.5); // second center -> index 1

    const result = kmeansPlusPlus(X, 2);
    expect(result.indices).toEqual([0, 1]);
    expect(result.centers).toEqual([X[0], X[1]]);

    mock.mockRestore();
    Math.random = originalRandom;
});

test('kmeans++ sample weight', () => {
    const X = [
        [0, 0],
        [1, 0],
        [10, 0]
    ];
    const weights = [10, 1, 1];
    const mock = jest
        .spyOn(Math, 'random')
        .mockReturnValueOnce(0.1) // because of weights, select index 0
        .mockReturnValueOnce(0.5); // choose farthest

    const result = kmeansPlusPlus(X, 2, weights);
    expect(result.indices).toEqual([0, 2]);
    mock.mockRestore();
});

test('kmeans++ identical points still returns n_clusters centers', () => {
    const X = [
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2]
    ];
    const mock = jest.spyOn(Math, 'random').mockReturnValue(0.3);

    const result = kmeansPlusPlus(X, 3);
    expect(result.centers).toHaveLength(3);
    expect(result.indices).toHaveLength(3);
    for (const center of result.centers) {
        expect(center).toEqual([1, 2]);
    }

    mock.mockRestore();
});

test('kmeans++ never selects zero-weight points', () => {
    const X = [
        [0, 0],
        [1, 0],
        [10, 0]
    ];
    const weights = [0, 0, 1];
    // randomState always returns 0: with the old `r <= cum` boundary this
    // selected index 0, whose probability mass is zero.
    const result = kmeansPlusPlus(X, 2, weights, () => 0);
    expect(result.indices).toEqual([2, 2]);
    expect(result.centers).toEqual([X[2], X[2]]);
});
