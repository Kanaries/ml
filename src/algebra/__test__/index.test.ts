import { gaussianElimination, Inverse } from '../index';
test('basic gussian elimination', () => {
    const A = [
        [0, -1, -1, 1],
        [1, 1, 1, 1],
        [2, 4, 1, -2],
        [3, 1, -2, 2]
    ]
    const Y = [[0], [6], [-1], [3]];
    const result = gaussianElimination(A, Y) as number[][];
    expect(result.length > 0).toBe(true);
    const ans = [
        [1, 1, 1, 1, 6],
        [0, 1, 1, -1, 0],
        [0, 0, 1, 2/3, 13/3],
        [0, 0, 0, 1, 2]
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