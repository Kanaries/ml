import { CSRMatrix } from '../data';

export function murmurHash3(value: string, seed = 0): number {
    const bytes = new TextEncoder().encode(value); let hash = seed | 0, offset = 0;
    while (offset + 4 <= bytes.length) {
        let k = (bytes[offset] | bytes[offset + 1] << 8 | bytes[offset + 2] << 16 | bytes[offset + 3] << 24) | 0; offset += 4;
        k = Math.imul(k, 0xcc9e2d51); k = k << 15 | k >>> 17; k = Math.imul(k, 0x1b873593);
        hash ^= k; hash = hash << 13 | hash >>> 19; hash = (Math.imul(hash, 5) + 0xe6546b64) | 0;
    }
    let k = 0;
    switch (bytes.length - offset) {
        case 3: k ^= bytes[offset + 2] << 16;
        case 2: k ^= bytes[offset + 1] << 8;
        case 1: k ^= bytes[offset]; k = Math.imul(k, 0xcc9e2d51); k = k << 15 | k >>> 17; k = Math.imul(k, 0x1b873593); hash ^= k;
    }
    hash ^= bytes.length; hash ^= hash >>> 16; hash = Math.imul(hash, 0x85ebca6b); hash ^= hash >>> 13; hash = Math.imul(hash, 0xc2b2ae35); hash ^= hash >>> 16;
    return hash | 0;
}

export function hashedRows(rows: Array<Array<[string, number]>>, nFeatures: number, alternateSign: boolean): CSRMatrix {
    const data: number[] = [], indices: number[] = [], indptr = [0];
    for (const row of rows) {
        const values = new Map<number, number>();
        for (const [feature, raw] of row) {
            const hash = murmurHash3(feature);
            const magnitude = hash === -2147483648 ? 2147483648 : Math.abs(hash);
            const index = magnitude % nFeatures, value = raw * (alternateSign && hash < 0 ? -1 : 1);
            values.set(index, (values.get(index) ?? 0) + value);
        }
        for (const [index, value] of Array.from(values).filter(([, value]) => value !== 0).sort((a, b) => a[0] - b[0])) { indices.push(index); data.push(value); }
        indptr.push(data.length);
    }
    return new CSRMatrix(data, indices, indptr, [rows.length, nFeatures]);
}
