import { assert } from "../utils";

export function valuesAllSame(arr: any[]): boolean {
    if (arr.length === 0) return true;
    const set = new Set();
    set.add(arr[0]);
    
    let i = 1;
    for (i = 1; i < arr.length; i++) {
        if (!set.has(arr[i])) return false;
    }
    return set.size === 1 ? true : false;
}
type ISplitMethod = 'random' | 'uniform';
export function generateSplitValues (min: number, max: number, number: number = 2, method: ISplitMethod = 'random'): number[] {
    assert(number >= 1, 'number should larger than 0');
    if (number === 1) return [];
    if (method === 'random') {
        const splitValues: number[] = new Array(number - 1).fill(0).map(() => Math.random());
        // 有一定概率存在相同的值，有待确定这种情况是否合理.
        splitValues.sort((a, b) => a - b);
        return splitValues;
    } else {
        const step = (max - min) / number;
        const splitValues: number[] = new Array(number - 1).fill(0).map((_, i) => min + (i + 1) * step);
        return splitValues;
    }
}

export function filterWithIndices<T = any>(arr: T[], condition: (v: T, i: number, arr: T[]) => boolean) {
    const subArr: T[] = [];
    const indices: number[] = [];
    for (let i = 0; i < arr.length; i++) {
        if (condition(arr[i], i, arr)) {
            subArr.push(arr[i]);
            indices.push(i)
        }
    }
    return {
        subArr,
        indices
    }
}

export function getUniqueFreqs(arr: any[]): number[] {
    const freqMap: Map<any, number> = new Map();
    for (let item of arr) {
        if (!freqMap.has(item)) {
            freqMap.set(item, 0);
        }
        freqMap.set(item, freqMap.get(item) + 1)
    }
    return [...freqMap.values()]
}