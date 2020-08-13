import { gaussianElimination } from '../index';
test('basic gussian elimination', () => {
    const A = [
        [0, -1, -1, 1],
        [1, 1, 1, 1],
        [2, 4, 1, -2],
        [3, 1, -2, 2]
    ]
    const Y = [0, 6, -1, 3];
    const result = gaussianElimination(A, Y);
    console.log(result)
    expect(result.length > 0).toBe(true);
    const ans = [
        [1, 1, 1, 1, 6],
        [0, 1, 1, -1, 0],
        [0, 0, 1, 2/3, 13/3],
        [0, 0, 0, 1, 2]
    ];
    // expect(result).toEqual(ans)
    for (let i = 0 ; i < A.length; i++) {
        for (let j = 0 ; j < A[i].length; j++) {
            expect(result[i][j]).toBeCloseTo(ans[i][j])
        }
    }
})