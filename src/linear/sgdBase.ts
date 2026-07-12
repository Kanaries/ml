/**
 * Shared core for the plain (non-averaged) stochastic gradient descent
 * estimators (SGDClassifier, SGDRegressor, Perceptron).
 *
 * Faithful port of scikit-learn's `_plain_sgd` (sklearn/linear_model/_sgd_fast.pyx):
 *  - per-sample updates over seeded-shuffled epochs;
 *  - learning-rate schedules `constant`, `optimal` (Léon Bottou's t0
 *    heuristic, exactly as in sklearn's `_init_t`), `invscaling`, `adaptive`;
 *  - L2 regularization by per-step weight scaling (intercept unregularized);
 *  - L1 / elastic-net via the cumulative-penalty trick (Tsuruoka et al. 2009);
 *  - convergence: stop when epoch loss > best - tol for nIterNoChange
 *    consecutive epochs (for `adaptive`, divide eta by 5 instead until
 *    eta <= 1e-6).
 */

export type SGDPenalty = 'l2' | 'l1' | 'elasticnet' | null;
export type SGDLearningRate = 'constant' | 'optimal' | 'invscaling' | 'adaptive';

/**
 * A pointwise loss. `p` is the model output (decision value / prediction),
 * `y` the target (±1 for classification losses, real for regression losses).
 * `dloss` is the (sub)gradient of the loss with respect to `p`.
 */
export interface SGDLoss {
    loss(p: number, y: number): number;
    dloss(p: number, y: number): number;
}

// ---------------------------------------------------------------------------
// Classification losses (targets in {-1, +1})
// ---------------------------------------------------------------------------

/** Hinge with configurable margin threshold: max(0, threshold - p*y). */
function hingeLoss(threshold: number): SGDLoss {
    return {
        loss: (p, y) => {
            const z = p * y;
            return z <= threshold ? threshold - z : 0;
        },
        dloss: (p, y) => (p * y <= threshold ? -y : 0),
    };
}

const squaredHingeLoss: SGDLoss = {
    loss: (p, y) => {
        const z = 1 - p * y;
        return z > 0 ? z * z : 0;
    },
    dloss: (p, y) => {
        const z = 1 - p * y;
        return z > 0 ? -2 * y * z : 0;
    },
};

/** Logistic regression loss, numerically stabilized like sklearn's `Log`. */
const logLoss: SGDLoss = {
    loss: (p, y) => {
        const z = p * y;
        if (z > 18) return Math.exp(-z);
        if (z < -18) return -z;
        return Math.log(1 + Math.exp(-z));
    },
    dloss: (p, y) => {
        const z = p * y;
        if (z > 18) return -y * Math.exp(-z);
        if (z < -18) return -y;
        return -y / (Math.exp(z) + 1);
    },
};

const modifiedHuberLoss: SGDLoss = {
    loss: (p, y) => {
        const z = p * y;
        if (z >= 1) return 0;
        if (z >= -1) return (1 - z) * (1 - z);
        return -4 * z;
    },
    dloss: (p, y) => {
        const z = p * y;
        if (z >= 1) return 0;
        if (z >= -1) return -2 * y * (1 - z);
        return -4 * y;
    },
};

export type ClassificationLossName = 'hinge' | 'logLoss' | 'modifiedHuber' | 'squaredHinge' | 'perceptron';

export function getClassificationLoss(name: ClassificationLossName): SGDLoss {
    switch (name) {
        case 'hinge': return hingeLoss(1);
        case 'perceptron': return hingeLoss(0);
        case 'squaredHinge': return squaredHingeLoss;
        case 'logLoss': return logLoss;
        case 'modifiedHuber': return modifiedHuberLoss;
        default:
            throw new Error(`Unknown classification loss "${name}". ` +
                "Valid losses are: 'hinge', 'logLoss', 'modifiedHuber', 'squaredHinge', 'perceptron'.");
    }
}

// ---------------------------------------------------------------------------
// Regression losses (real-valued targets)
// ---------------------------------------------------------------------------

const squaredErrorLoss: SGDLoss = {
    loss: (p, y) => 0.5 * (p - y) * (p - y),
    dloss: (p, y) => p - y,
};

function huberLoss(epsilon: number): SGDLoss {
    return {
        loss: (p, y) => {
            const r = p - y;
            const absR = Math.abs(r);
            return absR <= epsilon ? 0.5 * r * r : epsilon * absR - 0.5 * epsilon * epsilon;
        },
        dloss: (p, y) => {
            const r = p - y;
            if (Math.abs(r) <= epsilon) return r;
            return r > 0 ? epsilon : -epsilon;
        },
    };
}

function epsilonInsensitiveLoss(epsilon: number): SGDLoss {
    return {
        loss: (p, y) => {
            const z = Math.abs(y - p) - epsilon;
            return z > 0 ? z : 0;
        },
        dloss: (p, y) => {
            if (y - p > epsilon) return -1;
            if (p - y > epsilon) return 1;
            return 0;
        },
    };
}

function squaredEpsilonInsensitiveLoss(epsilon: number): SGDLoss {
    return {
        loss: (p, y) => {
            const z = Math.abs(y - p) - epsilon;
            return z > 0 ? z * z : 0;
        },
        dloss: (p, y) => {
            const z = y - p;
            if (z > epsilon) return -2 * (z - epsilon);
            if (-z > epsilon) return 2 * (-z - epsilon);
            return 0;
        },
    };
}

export type RegressionLossName = 'squaredError' | 'huber' | 'epsilonInsensitive' | 'squaredEpsilonInsensitive';

export function getRegressionLoss(name: RegressionLossName, epsilon: number): SGDLoss {
    switch (name) {
        case 'squaredError': return squaredErrorLoss;
        case 'huber': return huberLoss(epsilon);
        case 'epsilonInsensitive': return epsilonInsensitiveLoss(epsilon);
        case 'squaredEpsilonInsensitive': return squaredEpsilonInsensitiveLoss(epsilon);
        default:
            throw new Error(`Unknown regression loss "${name}". ` +
                "Valid losses are: 'squaredError', 'huber', 'epsilonInsensitive', 'squaredEpsilonInsensitive'.");
    }
}

// ---------------------------------------------------------------------------
// Core optimizer
// ---------------------------------------------------------------------------

/** Clip extreme gradients like sklearn's MAX_DLOSS guard. */
const MAX_DLOSS = 1e12;

export interface PlainSGDConfig {
    X: number[][];
    /** Targets: ±1 for classification losses, real values for regression. */
    y: number[];
    loss: SGDLoss;
    penalty: SGDPenalty;
    alpha: number;
    l1Ratio: number;
    fitIntercept: boolean;
    maxIter: number;
    /** Pass null to disable the convergence check (always run maxIter epochs). */
    tol: number | null;
    shuffle: boolean;
    /** Fit-local RNG in [0, 1). */
    rand: () => number;
    learningRate: SGDLearningRate;
    eta0: number;
    powerT: number;
    nIterNoChange: number;
}

export interface PlainSGDResult {
    weights: number[];
    intercept: number;
    /** Number of epochs actually run. */
    nIter: number;
}

/** In-place Fisher-Yates shuffle driven by the supplied RNG. */
export function shuffleIndices(indices: number[], rand: () => number): void {
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

export function plainSGD(config: PlainSGDConfig): PlainSGDResult {
    const {
        X, y, loss, penalty, alpha, l1Ratio, fitIntercept,
        maxIter, tol, shuffle, rand, learningRate, eta0, powerT, nIterNoChange,
    } = config;
    const nSamples = X.length;
    const nFeatures = X[0].length;
    const w = new Array<number>(nFeatures).fill(0);
    let intercept = 0;

    // Effective L1 and L2 mixing factors of the chosen penalty.
    const l1Factor = penalty === 'l1' ? 1 : penalty === 'elasticnet' ? l1Ratio : 0;
    const l2Factor = penalty === 'l2' ? 1 : penalty === 'elasticnet' ? 1 - l1Ratio : 0;

    // Cumulative L1 penalty bookkeeping (Tsuruoka et al. 2009), as in sklearn.
    const q = new Array<number>(nFeatures).fill(0);
    let u = 0;

    // Léon Bottou's t0 heuristic for the 'optimal' schedule — sklearn _init_t:
    //   typw = sqrt(1 / sqrt(alpha))
    //   initialEta0 = typw / max(1, dloss(-typw, 1))
    //   t0 = 1 / (initialEta0 * alpha)
    // then eta(t) = 1 / (alpha * (t0 + t - 1)) with t counting samples from 1.
    let optimalInit = 0;
    if (learningRate === 'optimal') {
        const typw = Math.sqrt(1 / Math.sqrt(alpha));
        const initialEta0 = typw / Math.max(1, loss.dloss(-typw, 1));
        optimalInit = 1 / (initialEta0 * alpha);
    }

    let eta = eta0; // used directly by 'constant' and 'adaptive'
    let t = 1;
    let bestLoss = Infinity;
    let noImprovementCount = 0;
    let nIter = 0;

    const indices = Array.from({ length: nSamples }, (_, i) => i);

    for (let epoch = 0; epoch < maxIter; epoch++) {
        nIter = epoch + 1;
        if (shuffle) shuffleIndices(indices, rand);
        let sumLoss = 0;
        for (let k = 0; k < nSamples; k++) {
            const i = indices[k];
            if (learningRate === 'optimal') {
                eta = 1 / (alpha * (optimalInit + t - 1));
            } else if (learningRate === 'invscaling') {
                eta = eta0 / Math.pow(t, powerT);
            }
            const x = X[i];
            let p = intercept;
            for (let j = 0; j < nFeatures; j++) p += w[j] * x[j];
            sumLoss += loss.loss(p, y[i]);
            let dl = loss.dloss(p, y[i]);
            if (dl > MAX_DLOSS) dl = MAX_DLOSS;
            else if (dl < -MAX_DLOSS) dl = -MAX_DLOSS;

            // L2: scale the weight vector (the intercept is never regularized).
            if (l2Factor > 0) {
                const scale = 1 - eta * alpha * l2Factor;
                if (scale <= 0) {
                    throw new Error('Learning rate too large: the L2 update scaling factor became non-positive. ' +
                        'Decrease eta0 or alpha.');
                }
                for (let j = 0; j < nFeatures; j++) w[j] *= scale;
            }

            // Gradient step.
            if (dl !== 0) {
                for (let j = 0; j < nFeatures; j++) w[j] -= eta * dl * x[j];
                if (fitIntercept) intercept -= eta * dl;
            }

            // Cumulative L1 penalty: clip each weight toward 0 by the total
            // L1 penalty it should have received, minus what it already got.
            if (l1Factor > 0) {
                u += l1Factor * eta * alpha;
                for (let j = 0; j < nFeatures; j++) {
                    const z = w[j];
                    if (z > 0) {
                        w[j] = Math.max(0, z - (u + q[j]));
                    } else if (z < 0) {
                        w[j] = Math.min(0, z + (u - q[j]));
                    }
                    q[j] += w[j] - z;
                }
            }
            t += 1;
        }

        // Convergence check on the average epoch (data) loss — sklearn semantics.
        const epochLoss = sumLoss / nSamples;
        if (tol !== null) {
            if (epochLoss > bestLoss - tol) noImprovementCount += 1;
            else noImprovementCount = 0;
        }
        if (epochLoss < bestLoss) bestLoss = epochLoss;
        if (noImprovementCount >= nIterNoChange) {
            if (learningRate === 'adaptive' && eta > 1e-6) {
                eta /= 5;
                noImprovementCount = 0;
            } else {
                break;
            }
        }
    }

    return { weights: w, intercept, nIter };
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

export function validateSGDData(X: number[][], y: number[]): void {
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must have at least one feature');
    }
    for (const row of X) {
        if (row.length !== nFeatures) {
            throw new Error('all rows in X must have the same length');
        }
    }
}

export function validatePenalty(penalty: unknown): SGDPenalty {
    if (penalty === null || penalty === 'l2' || penalty === 'l1' || penalty === 'elasticnet') {
        return penalty as SGDPenalty;
    }
    throw new Error(`Unknown penalty "${String(penalty)}". Valid penalties are: 'l2', 'l1', 'elasticnet', null.`);
}

export function validateSchedule(learningRate: SGDLearningRate, eta0: number, alpha: number): void {
    if (learningRate !== 'constant' && learningRate !== 'optimal'
        && learningRate !== 'invscaling' && learningRate !== 'adaptive') {
        throw new Error(`Unknown learning rate schedule "${String(learningRate)}". ` +
            "Valid schedules are: 'constant', 'optimal', 'invscaling', 'adaptive'.");
    }
    if (learningRate !== 'optimal' && eta0 <= 0) {
        throw new Error(`eta0 must be > 0 with learningRate='${learningRate}'`);
    }
    if (learningRate === 'optimal' && alpha <= 0) {
        throw new Error("alpha must be > 0 with learningRate='optimal' (alpha sets the schedule)");
    }
}
