import { resolveSubsetSize } from '../paramResolvers';

describe('resolveSubsetSize', () => {
    test('undefined and all mean the full set', () => {
        expect(resolveSubsetSize(undefined, 7)).toBe(7);
        expect(resolveSubsetSize('all', 7)).toBe(7);
    });

    test('sqrt/log2 floor like sklearn', () => {
        expect(resolveSubsetSize('sqrt', 5)).toBe(2); // sklearn int(sqrt(5)) = 2, not ceil = 3
        expect(resolveSubsetSize('sqrt', 9)).toBe(3);
        expect(resolveSubsetSize('log2', 9)).toBe(3);
        expect(resolveSubsetSize('sqrt', 1)).toBe(1);
        expect(resolveSubsetSize('log2', 1)).toBe(1); // floor(log2(1)) = 0 -> clamped to 1
    });

    test('fractions floor with a minimum of 1', () => {
        expect(resolveSubsetSize(0.5, 4)).toBe(2);
        expect(resolveSubsetSize(0.5, 5)).toBe(2);
        expect(resolveSubsetSize(0.1, 5)).toBe(1); // never 0
        expect(resolveSubsetSize(0.8, 10)).toBe(8);
    });

    test('integers are absolute counts, capped at total', () => {
        expect(resolveSubsetSize(1, 5)).toBe(1); // 1 = one feature, NOT 100%
        expect(resolveSubsetSize(3, 5)).toBe(3);
        expect(resolveSubsetSize(99, 5)).toBe(5);
    });

    test('invalid values throw', () => {
        expect(() => resolveSubsetSize(0, 5)).toThrow();
        expect(() => resolveSubsetSize(-1, 5)).toThrow();
        expect(() => resolveSubsetSize(NaN, 5)).toThrow();
        expect(() => resolveSubsetSize(1.5, 5)).toThrow();
        expect(() => resolveSubsetSize('sqr' as any, 5)).toThrow();
    });
});
