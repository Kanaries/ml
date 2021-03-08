import { assert } from "../utils";

export function cofactorMatrix (matrix: number[][], minorRowIndex: number, minorColIndex: number): number[][] {
    let ans: number[][] = [];
    for (let i = 0; i < matrix.length; i++) {
        if (i === minorRowIndex) continue;
        const row = [];
        for (let j = 0; j < matrix[i].length; j++) {
            if (j === minorColIndex) continue;
            row.push(matrix[i][j]);
        }
        ans.push(row);
    }
    return ans;
}

export function det(matrix: number[][]) {
    assert(matrix.length > 0, 'matrix should not be empty');
    assert(matrix[0].length === matrix.length, 'matrix should be an n X n matrix.')
    const n = matrix.length;
    if (n === 1) return matrix[0][0];
    let ans = 0;
    // op = Math.pow(-1, i + j)
    for (let i = 0, op = 1; i < n; i++, op *= -1) {
        const cofactor = op * det(cofactorMatrix(matrix, 0, i));
        ans += matrix[0][i] * cofactor;
    }
    return ans;
}