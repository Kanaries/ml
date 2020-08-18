import { gaussianElimination, identityMatrix, augmentMatrix } from "./basic";

export function elementary (A: number[][]): number[][] | false {
    const I = identityMatrix(A);
    const originAM = augmentMatrix(A, I);
    const AM: number[][] | false = gaussianElimination(originAM);
    if (AM === false) return false;
    const size = A.length;
    for (let i = size - 1; i >= 0; i--) {
        for (let j = i - 1; j >= 0; j--) {
            if (AM[j][i] !== 0) {
                const multi = AM[j][i] / AM[i][i];
                for (let k = 0; k < AM[j].length; k++) {
                    AM[j][k] -= AM[i][k] * multi;
                }
            }
        }
        const normalizer = AM[i][i];
        for (let j = 0; j < AM[i].length; j++) {
            AM[i][j] /= normalizer;
        }
    }
    let A_Inverse: number[][] = [];
    for (let i = 0; i < size; i++) {
        A_Inverse.push([])
        for (let j = size; j < AM[i].length; j++) {
            A_Inverse[i].push(AM[i][j]);
        }
    }
    return A_Inverse;
}