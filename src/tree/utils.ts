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