import { resolveDistanceWeights, weightedMode } from '../utils';

test('resolveDistanceWeights inverts non-zero distances', () => {
    expect(resolveDistanceWeights([12, 3, 5, 8])).toEqual([1 / 12, 1 / 3, 1 / 5, 1 / 8]);
});

test('resolveDistanceWeights gives all weight to zero-distance samples', () => {
    expect(resolveDistanceWeights([0, 2, 0, 8])).toEqual([1, 0, 1, 0]);
});

test('weightedMode picks the label with the largest total weight', () => {
    expect(weightedMode([1, 2, 3, 4, 5, 5], [1, 1, 1, 1, 1, 1])).toEqual(5);
    expect(weightedMode([1, 2, 3, 4, 5, 5], [100, 1, 1, 1, 1, 1])).toEqual(1);
});

test('weightedMode resolves ties to the smallest label', () => {
    expect(weightedMode([5, 1], [1, 1])).toEqual(1);
    expect(weightedMode([9, 3, 9, 3], [1, 1, 1, 1])).toEqual(3);
});
