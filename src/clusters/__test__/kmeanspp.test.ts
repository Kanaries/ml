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
