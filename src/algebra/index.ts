export function gaussianElimination (A: number[][], Y: number[]): number[][] {
    let augmentedMatrix: number[][] = [];
    for (let i = 0; i < A.length; i++) {
        augmentedMatrix.push([...A[i], Y[i]])
    }
    for (let i = 0; i < augmentedMatrix.length; i++) {
        if (augmentedMatrix[i][i] === 0) {
            for (let j = i + 1; j < augmentedMatrix.length; j++) {
                if (augmentedMatrix[j][i] !== 0) {
                    let t = augmentedMatrix[j];
                    augmentedMatrix[j] = augmentedMatrix[i];
                    augmentedMatrix[i] = t;
                    break;
                }
            }
        }
        for (let j = i + 1; j < augmentedMatrix.length; j++) {
            const multi = augmentedMatrix[j][i] / augmentedMatrix[i][i];
            for (let k = 0; k < augmentedMatrix[j].length; k++) {
                augmentedMatrix[j][k] -= augmentedMatrix[i][k] * multi;
            }
        }
        const normalizer = augmentedMatrix[i][i];
        for (let j = 0; j < augmentedMatrix[i].length; j++) {
            console.log(augmentedMatrix[i][j], augmentedMatrix[i][i]);
            augmentedMatrix[i][j] /= normalizer;
        }
    }
    return augmentedMatrix;
}