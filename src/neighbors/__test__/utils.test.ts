import { getWeights, votes } from '../utils';

test('getWeights', () => {
    // @ts-ignore
    expect(() => getWeights([], 'wrongtype')).toThrowError();
    expect(getWeights([12, 3, 5, 8], 'uniform')).toEqual([1, 1, 1, 1]);
    expect(getWeights([12, 3, 5, 8], 'distance')).toEqual([1 / 12, 1 / 3, 1 / 5, 1 / 8]);
});

test('votes', () => {
    expect(votes([1, 2, 3, 4, 5, 5], [1, 1, 1, 1, 1, 1])).toEqual(5);
    expect(votes([1, 2, 3, 4, 5, 5], [100, 1, 1, 1, 1, 1])).toEqual(1);
});
