export function std<T = any> (arr: T[], size: number): T[] {
    const n = arr.length;
    const k = Math.floor(size);
    if (n === 0 || k <= 0) {
        return [];
    }
    if (k >= n) {
        return arr.slice();
    }

    // Reservoir sampling (sampling without replacement).
    const sample = arr.slice(0, k);
    for (let i = k; i < n; i++) {
        const j = Math.floor(Math.random() * (i + 1));
        if (j < k) {
            sample[j] = arr[i];
        }
    }
    return sample;
}
