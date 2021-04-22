import { gaussianElimination, Inverse } from '../index';
import { transpose, augmentMatrix, product } from '../basic';
test('basic gussian elimination', () => {
    const A = [
        [0, -1, -1, 1],
        [1, 1, 1, 1],
        [2, 4, 1, -2],
        [3, 1, -2, 2]
    ]
    const Y = [[0], [6], [-1], [3]];
    const AM = augmentMatrix(A, Y);
    const result = gaussianElimination(AM) as number[][];
    expect(result instanceof Array).toBe(true)
    expect(result.length > 0).toBe(true);
    // const ans = [
    //     [1, 1, 1, 1, 6],
    //     [0, 1, 1, -1, 0],
    //     [0, 0, 1, 2/3, 13/3],
    //     [0, 0, 0, 1, 2]
    // ];
    const ans = [
        [1, 1, 1, 1],
        [0, 1, 1, -1],
        [0, 0, 1, 2 / 3],
        [0, 0, 0, 1],
    ];
    for (let i = 0 ; i < A.length; i++) {
        for (let j = 0 ; j < A[i].length; j++) {
            expect(result[i][j]).toBeCloseTo(ans[i][j])
        }
    }
})

test('inverse of matrix', () => {
    const A = [
        [1, 4, 3],
        [-1, -2, 0],
        [2, 2, 3]
    ];
    const A_Inverse = Inverse.elementary(A) as number[][];
    const ans = [
        [-1/2, -1/2, 1/2],
        [1/4, -1/4, -1/4],
        [1/6, 1/2, 1/6]
    ];
    for (let i = 0; i < ans.length; i++) {
        for (let j = 0; j < ans[i].length; j++) {
            expect(A_Inverse[i][j]).toBeCloseTo(ans[i][j]);
        }
    }
})

test('transpose', () => {
    const width = 10;
    const height = 20;
    const matrix: number[][] = [];
    for (let i = 0; i < height; i++) {
        matrix.push([])
        for (let j = 0; j < width; j++) {
            matrix[i].push(Math.round(100 * Math.random()));
        }
    }
    const matrix_t = transpose(matrix);
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            expect(matrix[i][j]).toBe(matrix_t[j][i])
        }
    }
})

test('product', () => {
    const A = [
        [3, -2],
        [2, 4],
        [1, -3]
    ];
    const B = [
        [-2, 1, 3],
        [4, 1, 6]
    ];
    const result = product(A, B);
    const ans = [
        [-14, 1, -3],
        [12, 6, 30],
        [-14, -2, -15]
    ]
    expect(result).toEqual(ans);
})