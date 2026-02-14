import { std } from '../sampling';

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
