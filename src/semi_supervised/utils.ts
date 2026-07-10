export function argMax(arr: number[]): number {
    let m = -Infinity;
    let idx = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > m) {
            m = arr[i];
            idx = i;
        }
    }
    return idx;
}

export function rowNormalize(mat: number[][]): number[][] {
    return mat.map(row => {
        const s = row.reduce((a, b) => a + b, 0);
        if (s === 0) return row.map(() => 0);
        return row.map(v => v / s);
    });
}

export function multiply(A: number[][], B: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < A.length; i++) {
        const row: number[] = new Array(B[0].length).fill(0);
        for (let k = 0; k < B.length; k++) {
            for (let j = 0; j < B[0].length; j++) {
                row[j] += A[i][k] * B[k][j];
            }
        }
        result.push(row);
    }
    return result;
}

export function transpose(A: number[][]): number[][] {
    const result: number[][] = [];
    for (let j = 0; j < A[0].length; j++) {
        const row: number[] = [];
        for (let i = 0; i < A.length; i++) {
            row.push(A[i][j]);
        }
        result.push(row);
    }
    return result;
}

export function rbfKernel(X: number[][], Y: number[][], gamma: number): number[][] {
    const result: number[][] = [];
    for (const x of X) {
        const row: number[] = [];
        for (const y of Y) {
            let d = 0;
            for (let i = 0; i < x.length; i++) {
                const diff = x[i] - y[i];
                d += diff * diff;
            }
            row.push(Math.exp(-gamma * d));
        }
        result.push(row);
    }
    return result;
}

export function knnKernel(X: number[][], Y: number[][], nNeighbors: number): number[][] {
    const result: number[][] = [];
    for (const x of X) {
        const dists: { d: number; i: number }[] = [];
        for (let i = 0; i < Y.length; i++) {
            const y = Y[i];
            let d = 0;
            for (let j = 0; j < x.length; j++) {
                const diff = x[j] - y[j];
                d += diff * diff;
            }
            dists.push({ d: Math.sqrt(d), i });
        }
        dists.sort((a, b) => a.d - b.d);
        const row = new Array(Y.length).fill(0);
        for (let k = 0; k < Math.min(nNeighbors, dists.length); k++) {
            row[dists[k].i] = 1;
        }
        result.push(row);
    }
    return result;
}
