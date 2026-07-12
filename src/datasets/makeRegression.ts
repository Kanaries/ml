import {
    createRandomGenerator,
    createGaussianGenerator,
    gaussianMatrix,
    matMul,
    permutation,
} from './common';

export interface MakeRegressionProps {
    /** number of samples */
    nSamples?: number;
    /** number of features */
    nFeatures?: number;
    /** number of features actually used to build the linear model */
    nInformative?: number;
    /** number of regression targets */
    nTargets?: number;
    /** bias term of the underlying linear model */
    bias?: number;
    /**
     * approximate number of singular vectors required to explain most of
     * the input variance; when set, X becomes a low-rank matrix with a
     * bell-shaped singular value profile instead of i.i.d. gaussian noise
     */
    effectiveRank?: number;
    /** relative importance of the fat noisy tail of the singular profile (with effectiveRank) */
    tailStrength?: number;
    /** standard deviation of the gaussian noise added to the targets */
    noise?: number;
    /** shuffle the samples and the feature columns (default true) */
    shuffle?: boolean;
    /** also return the ground-truth coefficients */
    coef?: boolean;
    /** seed for reproducible output */
    randomState?: number;
}

export interface MakeRegressionResult {
    /** samples, shape [nSamples][nFeatures] */
    X: number[][];
    /** targets: number[] when nTargets === 1, number[][] ([nSamples][nTargets]) otherwise */
    y: number[] | number[][];
    /**
     * ground-truth coefficients (only when `coef: true`): number[] of
     * length nFeatures when nTargets === 1, number[][] ([nFeatures][nTargets]) otherwise
     */
    coef?: number[] | number[][];
}

/**
 * Generates a random regression problem, y = X . coef + bias + noise
 * (port of sklearn.datasets.make_regression). Only nInformative features
 * have non-zero coefficients (uniform in [0, 100)).
 *
 * `effectiveRank` is fully implemented (sklearn's make_low_rank_matrix):
 * X = U diag(s) V^T with orthonormal U, V from QR of gaussian matrices and
 * a bell-shaped + fat-tail singular value profile controlled by tailStrength.
 */
export function makeRegression(props: MakeRegressionProps = {}): MakeRegressionResult {
    const {
        nSamples = 100,
        nFeatures = 100,
        nTargets = 1,
        bias = 0,
        effectiveRank,
        tailStrength = 0.5,
        noise = 0,
        shuffle = true,
        coef = false,
        randomState,
    } = props;
    const nInformative = Math.min(nFeatures, props.nInformative ?? 10);

    const rng = createRandomGenerator(randomState);
    const gaussian = createGaussianGenerator(rng);

    let X: number[][];
    if (effectiveRank === undefined) {
        // randomly generated features, well conditioned, centered, gaussian
        X = gaussianMatrix(nSamples, nFeatures, gaussian);
    } else {
        if (effectiveRank <= 0) {
            throw new Error('effectiveRank must be a positive number');
        }
        X = makeLowRankMatrix(nSamples, nFeatures, effectiveRank, tailStrength, gaussian);
    }

    // ground truth model, with only nInformative non-zero coefficient rows
    const groundTruth: number[][] = [];
    for (let j = 0; j < nFeatures; j++) {
        groundTruth.push(new Array(nTargets).fill(0));
    }
    for (let j = 0; j < nInformative; j++) {
        for (let t = 0; t < nTargets; t++) {
            groundTruth[j][t] = 100 * rng();
        }
    }

    const Y: number[][] = matMul(X, groundTruth).map((row) => row.map((value) => value + bias));
    if (noise > 0) {
        for (let i = 0; i < nSamples; i++) {
            for (let t = 0; t < nTargets; t++) {
                Y[i][t] += noise * gaussian();
            }
        }
    }

    let outX = X;
    let outY = Y;
    let outCoef = groundTruth;
    if (shuffle) {
        const rowOrder = permutation(nSamples, rng);
        outX = rowOrder.map((i) => outX[i]);
        outY = rowOrder.map((i) => outY[i]);
        // permute the feature columns together with the coefficient rows
        const featureOrder = permutation(nFeatures, rng);
        outX = outX.map((row) => featureOrder.map((j) => row[j]));
        outCoef = featureOrder.map((j) => outCoef[j]);
    }

    const result: MakeRegressionResult = {
        X: outX,
        y: nTargets === 1 ? outY.map((row) => row[0]) : outY,
    };
    if (coef) {
        result.coef = nTargets === 1 ? outCoef.map((row) => row[0]) : outCoef;
    }
    return result;
}

/**
 * Mostly low-rank matrix with bell-shaped singular values
 * (port of sklearn.datasets.make_low_rank_matrix).
 */
export function makeLowRankMatrix(
    nSamples: number,
    nFeatures: number,
    effectiveRank: number,
    tailStrength: number,
    gaussian: () => number,
): number[][] {
    const n = Math.min(nSamples, nFeatures);

    // economic QR of two random gaussian matrices -> orthonormal U, V columns
    const u = orthonormalColumns(gaussianMatrix(nSamples, n, gaussian));
    const v = orthonormalColumns(gaussianMatrix(nFeatures, n, gaussian));

    // index of the singular values
    const s: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
        const singularIndex = i / effectiveRank;
        const lowRank = (1 - tailStrength) * Math.exp(-1.0 * singularIndex * singularIndex);
        const tail = tailStrength * Math.exp(-0.1 * singularIndex);
        s[i] = lowRank + tail;
    }

    // X = (U * s) . V^T
    const out: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
        const row: number[] = new Array(nFeatures).fill(0);
        for (let k = 0; k < n; k++) {
            const scaled = u[i][k] * s[k];
            for (let j = 0; j < nFeatures; j++) {
                row[j] += scaled * v[j][k];
            }
        }
        out.push(row);
    }
    return out;
}

/**
 * Orthonormalises the columns of `m` ([rows][cols], rows >= cols) with
 * modified Gram-Schmidt (the Q factor of an economic QR decomposition).
 */
function orthonormalColumns(m: number[][]): number[][] {
    const rows = m.length;
    const cols = m[0].length;
    // work column-major
    const q: number[][] = [];
    for (let j = 0; j < cols; j++) {
        const col: number[] = new Array(rows);
        for (let i = 0; i < rows; i++) {
            col[i] = m[i][j];
        }
        for (let p = 0; p < q.length; p++) {
            let dot = 0;
            for (let i = 0; i < rows; i++) {
                dot += col[i] * q[p][i];
            }
            for (let i = 0; i < rows; i++) {
                col[i] -= dot * q[p][i];
            }
        }
        let norm = 0;
        for (let i = 0; i < rows; i++) {
            norm += col[i] * col[i];
        }
        norm = Math.sqrt(norm);
        if (norm === 0) {
            throw new Error('rank-deficient random matrix in QR (should not happen with gaussian input)');
        }
        for (let i = 0; i < rows; i++) {
            col[i] /= norm;
        }
        q.push(col);
    }
    // back to row-major [rows][cols]
    const out: number[][] = [];
    for (let i = 0; i < rows; i++) {
        const row: number[] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = q[j][i];
        }
        out.push(row);
    }
    return out;
}
