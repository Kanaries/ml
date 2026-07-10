import { sum, expected, variance, varSample, varPopulation, SDSample, SDPopulation, covariance } from '../index';

describe('math basics', () => {
    const nums = [2, 4, 4, 4, 5, 5, 7, 9];

    test('sum and expected', () => {
        expect(sum(nums)).toBe(40);
        expect(expected(nums)).toBe(5);
    });

    test('variance divides sum of squared deviations by df', () => {
        // SS = 9+1+1+1+0+0+4+16 = 32
        expect(variance(nums, nums.length)).toBeCloseTo(4, 10);
        expect(variance(nums, nums.length - 1)).toBeCloseTo(32 / 7, 10);
    });

    test('varPopulation matches hand-computed value', () => {
        expect(varPopulation(nums)).toBeCloseTo(4, 10);
        expect(varPopulation([0, 2])).toBeCloseTo(1, 10);
    });

    test('varSample matches hand-computed value', () => {
        expect(varSample(nums)).toBeCloseTo(32 / 7, 10);
        expect(varSample([0, 2])).toBeCloseTo(2, 10);
    });

    test('standard deviations are square roots of the variances', () => {
        expect(SDPopulation(nums)).toBeCloseTo(2, 10);
        expect(SDSample(nums)).toBeCloseTo(Math.sqrt(32 / 7), 10);
    });

    test('covariance is population covariance', () => {
        expect(covariance([1, 2, 3], [2, 4, 6])).toBeCloseTo(4 / 3, 10);
    });
});
