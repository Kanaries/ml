import { std, trainTestSplit } from '../sampling';

test('std returns empty for non-positive size', () => {
    expect(std([1, 2, 3], 0)).toEqual([]);
    expect(std([1, 2, 3], -5)).toEqual([]);
});

test('std returns copy when requesting enough samples', () => {
    const data = [1, 2, 3];
    const sampled = std(data, 3);
    expect(sampled).toEqual(data);
    expect(sampled).not.toBe(data);
});

test('std sample size is respected without duplicates', () => {
    const data = [1, 2, 3, 4, 5];
    const sampled = std(data, 3);
    expect(sampled).toHaveLength(3);
    const uniqCount = new Set(sampled).size;
    expect(uniqCount).toBe(sampled.length);
});

test('trainTestSplit returns deterministic split with randomState', () => {
    const X = [1, 2, 3, 4, 5, 6];
    const y = ['a', 'b', 'c', 'd', 'e', 'f'];
    const run1 = trainTestSplit(X, y, { testSize: 2, randomState: 42 });
    const run2 = trainTestSplit(X, y, { testSize: 2, randomState: 42 });

    expect(run1).toEqual(run2);
    expect(run1.XTrain).toHaveLength(4);
    expect(run1.XTest).toHaveLength(2);
    expect(run1.yTrain).toHaveLength(4);
    expect(run1.yTest).toHaveLength(2);
});

test('trainTestSplit supports non-shuffled split', () => {
    const X = [10, 20, 30, 40];
    const y = ['x', 'y', 'z', 'w'];
    const result = trainTestSplit(X, y, { testSize: 0.5, shuffle: false });

    expect(result.XTrain).toEqual([10, 20]);
    expect(result.XTest).toEqual([30, 40]);
    expect(result.yTrain).toEqual(['x', 'y']);
    expect(result.yTest).toEqual(['z', 'w']);
});

test('trainTestSplit works without labels', () => {
    const X = [[1], [2], [3], [4]];
    const result = trainTestSplit(X, undefined, { testSize: 1, shuffle: false });

    expect(result.XTrain).toEqual([[1], [2], [3]]);
    expect(result.XTest).toEqual([[4]]);
    expect(result.yTrain).toBeUndefined();
    expect(result.yTest).toBeUndefined();
});

test('trainTestSplit validates X/y shape mismatch', () => {
    expect(() => trainTestSplit([1, 2, 3], [1, 2], { testSize: 1 })).toThrow('X and y must have the same length');
});

test('trainTestSplit validates illegal testSize', () => {
    expect(() => trainTestSplit([1, 2, 3], undefined, { testSize: 0 })).toThrow('testSize must be a positive finite number');
    expect(() => trainTestSplit([1, 2, 3], undefined, { testSize: 3 })).toThrow('testSize must produce at least 1 test sample and leave at least 1 train sample');
});
