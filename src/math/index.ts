import { assert } from "../utils";

export function sum(nums: number[]): number {
    let s = 0;
    for (let i = 0; i < nums.length; i++) {
        s += nums[i];
    }
    return s;
}

export function expected(nums: number[]): number {
    let s = sum(nums);
    return s / nums.length;
}

export function variance (nums: number[], df: number): number {
    const mu = expected(nums);
    return expected(nums.map(n => (n - mu) ** 2)) / df;
}

export function covariance(X: number[], Y: number[]): number {
    assert(X.length === Y.length, 'X and Y should have a same length.')
    const meanX = expected(X);
    const meanY = expected(Y);
    return expected(X.map((x, i) => (x - meanX) * (Y[i] - meanY)))
}

export function varSample(nums: number[]): number {
    return variance(nums, nums.length - 1);
}

export function SDSample(nums: number[]): number {
    return Math.sqrt(varSample(nums))
}

export function varPopulation(nums: number[]): number {
    return variance(nums, nums.length);
}

export function SDPopulation(nums: number[]): number {
    return Math.sqrt(varPopulation(nums));
}