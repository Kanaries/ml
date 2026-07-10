import { assert } from "../utils";

export function gaussianElimination(A: number[][]): number[][] | false {
    let rowEchelon: number[][] = [];
    let maxAbs = 0;
    for (let i = 0; i < A.length; i++) {
        rowEchelon.push([])
        for (let j = 0; j < A[i].length; j++) {
            rowEchelon[i].push(A[i][j])
            maxAbs = Math.max(maxAbs, Math.abs(A[i][j]));
        }
    }
    // pivots below this are numerically zero relative to the matrix scale;
    // eliminating with them would amplify float noise into garbage results
    const tol = maxAbs * A.length * Number.EPSILON;
    for (let i = 0; i < rowEchelon.length; i++) {
        // partial pivoting: bring the largest |value| in the column up so
        // tiny (or float-noise) pivots never divide the remaining rows
        let pivotRow = i;
        for (let j = i + 1; j < rowEchelon.length; j++) {
            if (Math.abs(rowEchelon[j][i]) > Math.abs(rowEchelon[pivotRow][i])) {
                pivotRow = j;
            }
        }
        if (Math.abs(rowEchelon[pivotRow][i]) <= tol) return false;
        if (pivotRow !== i) {
            const t = rowEchelon[pivotRow];
            rowEchelon[pivotRow] = rowEchelon[i];
            rowEchelon[i] = t;
        }
        for (let j = i + 1; j < rowEchelon.length; j++) {
            const multi = rowEchelon[j][i] / rowEchelon[i][i];
            for (let k = 0; k < rowEchelon[j].length; k++) {
                rowEchelon[j][k] -= rowEchelon[i][k] * multi;
            }
        }
        const normalizer = rowEchelon[i][i];
        for (let j = 0; j < rowEchelon[i].length; j++) {
            rowEchelon[i][j] /= normalizer;
        }
    }
    return rowEchelon;
}

export function identityMatrix(A: number[][]): number[][] {
    let size = A.length;
    let I: number[][] = [];
    for (let i = 0; i < size; i++) {
        I.push([]);
        for (let j = 0; j < size; j++) {
            I[i].push(i === j ? 1 : 0);
        }
    }
    return I;
}

export function transpose(matrix: number[][]): number[][] {
    assert(matrix.length > 0, 'matrix should be non-empty.')
    assert(matrix[0].length > 0, 'matrix should be non-empty.');
    let trans: number[][] = [];
    for (let i = 0; i < matrix[0].length; i++) {
        trans.push([])
    }
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            trans[j].push(matrix[i][j])
        }
    }
    return trans;
}

export function augmentMatrix(matrix: number[][], expanded: number[][]): number[][] {
    assert(matrix.length === expanded.length, 'submatrices should share the same height.')
    let AM: number[][] = [];
    for (let i = 0; i < matrix.length; i++) {
        AM.push([...matrix[i], ...expanded[i]])
    }
    return AM;
}
