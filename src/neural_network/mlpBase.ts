/**
 * Shared training core for MLPClassifier / MLPRegressor.
 *
 * Plain helper functions plus fit-local optimizer classes. Nothing here is
 * ever stored on an estimator instance (weights come back as plain
 * number[][][] / number[][]), so no registerSerializableClass is needed —
 * the same pattern xgboostBase.ts uses for its shared boosting loop.
 *
 * Mirrors scikit-learn's `MLPClassifier`/`MLPRegressor` stochastic path
 * (`_multilayer_perceptron._fit_stochastic`):
 *  - Glorot-style uniform init (`_init_coef`): bound =
 *    sqrt(6 / (fanIn + fanOut)) for identity/tanh/relu and
 *    sqrt(2 / (fanIn + fanOut)) for logistic; weights AND biases are drawn
 *    uniformly from [-bound, bound].
 *  - `batchSize: 'auto'` = min(200, nSamples); minibatches are contiguous
 *    slices of a per-epoch (optionally shuffled) index permutation.
 *  - Adam with bias-corrected first/second moments
 *    (lr_t = lrInit * sqrt(1 - beta2^t) / (1 - beta1^t), t per minibatch).
 *  - SGD with classical or Nesterov momentum and the constant / invscaling /
 *    adaptive schedules. invscaling: lr = lrInit / (t + 1)^powerT with t the
 *    number of samples seen. adaptive: lr divided by 5 whenever training
 *    stalls for nIterNoChange epochs, stopping once lr <= 1e-6.
 *  - Convergence: tol on training loss (or on validation score when
 *    earlyStopping) with an nIterNoChange patience window; training also
 *    stops silently at maxIter.
 *
 * The lbfgs solver is intentionally NOT implemented (only 'adam' and 'sgd').
 * Early-stopping validation split is a plain deterministic tail split (see
 * `trainMLP`), unlike sklearn's shuffled (and, for classifiers, stratified)
 * `train_test_split`.
 */
import { createRandomGenerator } from '../utils';

export type MLPActivation = 'identity' | 'logistic' | 'tanh' | 'relu';
export type MLPSolver = 'adam' | 'sgd';
export type MLPLearningRateSchedule = 'constant' | 'invscaling' | 'adaptive';
export type MLPOutActivation = 'identity' | 'logistic' | 'softmax';
export type MLPLossKind = 'log' | 'binaryLog' | 'squared';

export interface MLPProps {
    hiddenLayerSizes?: number[];
    activation?: MLPActivation;
    solver?: MLPSolver;
    alpha?: number;
    batchSize?: number | 'auto';
    learningRate?: MLPLearningRateSchedule;
    learningRateInit?: number;
    powerT?: number;
    maxIter?: number;
    shuffle?: boolean;
    randomState?: number;
    tol?: number;
    momentum?: number;
    nesterovsMomentum?: boolean;
    earlyStopping?: boolean;
    validationFraction?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
    nIterNoChange?: number;
}

/** Resolved, validated hyper-parameters (exactly the getParams() shape). */
export interface MLPParams {
    hiddenLayerSizes: number[];
    activation: MLPActivation;
    solver: MLPSolver;
    alpha: number;
    batchSize: number | 'auto';
    learningRate: MLPLearningRateSchedule;
    learningRateInit: number;
    powerT: number;
    maxIter: number;
    shuffle: boolean;
    randomState: number | undefined;
    tol: number;
    momentum: number;
    nesterovsMomentum: boolean;
    earlyStopping: boolean;
    validationFraction: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    nIterNoChange: number;
}

const ACTIVATIONS: readonly MLPActivation[] = ['identity', 'logistic', 'tanh', 'relu'];
const SOLVERS: readonly MLPSolver[] = ['adam', 'sgd'];
const SCHEDULES: readonly MLPLearningRateSchedule[] = ['constant', 'invscaling', 'adaptive'];

/** Resolve props to the canonical parameter set with sklearn's defaults. */
export function resolveMLPProps(props: MLPProps = {}): MLPParams {
    const params: MLPParams = {
        hiddenLayerSizes: [...(props.hiddenLayerSizes ?? [100])],
        activation: props.activation ?? 'relu',
        solver: props.solver ?? 'adam',
        alpha: props.alpha ?? 1e-4,
        batchSize: props.batchSize ?? 'auto',
        learningRate: props.learningRate ?? 'constant',
        learningRateInit: props.learningRateInit ?? 1e-3,
        powerT: props.powerT ?? 0.5,
        maxIter: props.maxIter ?? 200,
        shuffle: props.shuffle ?? true,
        randomState: props.randomState,
        tol: props.tol ?? 1e-4,
        momentum: props.momentum ?? 0.9,
        nesterovsMomentum: props.nesterovsMomentum ?? true,
        earlyStopping: props.earlyStopping ?? false,
        validationFraction: props.validationFraction ?? 0.1,
        beta1: props.beta1 ?? 0.9,
        beta2: props.beta2 ?? 0.999,
        epsilon: props.epsilon ?? 1e-8,
        nIterNoChange: props.nIterNoChange ?? 10,
    };

    if (params.hiddenLayerSizes.length === 0 ||
        params.hiddenLayerSizes.some(s => !Number.isInteger(s) || s < 1)) {
        throw new Error('hiddenLayerSizes must be a non-empty array of positive integers');
    }
    if (!ACTIVATIONS.includes(params.activation)) {
        throw new Error(`activation must be one of ${ACTIVATIONS.join(', ')}`);
    }
    if (!SOLVERS.includes(params.solver)) {
        throw new Error(`solver must be one of ${SOLVERS.join(', ')} (lbfgs is not supported)`);
    }
    if (!Number.isFinite(params.alpha) || params.alpha < 0) {
        throw new Error('alpha must be a non-negative finite number');
    }
    if (params.batchSize !== 'auto' && (!Number.isInteger(params.batchSize) || params.batchSize < 1)) {
        throw new Error("batchSize must be 'auto' or a positive integer");
    }
    if (!SCHEDULES.includes(params.learningRate)) {
        throw new Error(`learningRate must be one of ${SCHEDULES.join(', ')}`);
    }
    if (!(params.learningRateInit > 0)) {
        throw new Error('learningRateInit must be positive');
    }
    if (!Number.isFinite(params.powerT)) {
        throw new Error('powerT must be finite');
    }
    if (!Number.isInteger(params.maxIter) || params.maxIter < 1) {
        throw new Error('maxIter must be a positive integer');
    }
    if (!(params.tol > 0)) {
        throw new Error('tol must be positive');
    }
    if (!(params.momentum >= 0 && params.momentum <= 1)) {
        throw new Error('momentum must be in [0, 1]');
    }
    if (!(params.validationFraction > 0 && params.validationFraction < 1)) {
        throw new Error('validationFraction must be in (0, 1)');
    }
    if (!(params.beta1 >= 0 && params.beta1 < 1)) {
        throw new Error('beta1 must be in [0, 1)');
    }
    if (!(params.beta2 >= 0 && params.beta2 < 1)) {
        throw new Error('beta2 must be in [0, 1)');
    }
    if (!(params.epsilon > 0)) {
        throw new Error('epsilon must be positive');
    }
    if (!Number.isInteger(params.nIterNoChange) || params.nIterNoChange < 1) {
        throw new Error('nIterNoChange must be a positive integer');
    }
    return params;
}

export function validateMLPFitInput(X: number[][], y: ArrayLike<unknown>): void {
    if (X.length === 0) {
        throw new Error('X must be non-empty');
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
            throw new Error('all rows of X must have the same number of features');
        }
        for (const v of row) {
            if (!Number.isFinite(v)) {
                throw new Error('X contains NaN or non-finite values, which are not supported');
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weights: plain arrays, serializable as-is when stored on an estimator
// ---------------------------------------------------------------------------

export interface MLPWeights {
    /** coefs[l][i][j]: weight from unit i of layer l to unit j of layer l+1. */
    coefs: number[][][];
    intercepts: number[][];
}

/**
 * sklearn `_init_coef`: uniform in [-bound, bound] with
 * bound = sqrt(factor / (fanIn + fanOut)), factor 2 for logistic, else 6.
 * Weights are drawn row-major, then the biases — one RNG stream for the
 * whole net, so a fixed randomState fully determines the init.
 */
export function initMLPWeights(
    layerUnits: number[],
    activation: MLPActivation,
    rand: () => number
): MLPWeights {
    const factor = activation === 'logistic' ? 2 : 6;
    const coefs: number[][][] = [];
    const intercepts: number[][] = [];
    for (let l = 0; l < layerUnits.length - 1; l++) {
        const fanIn = layerUnits[l];
        const fanOut = layerUnits[l + 1];
        const bound = Math.sqrt(factor / (fanIn + fanOut));
        const coef: number[][] = [];
        for (let i = 0; i < fanIn; i++) {
            const row: number[] = new Array(fanOut);
            for (let j = 0; j < fanOut; j++) {
                row[j] = (rand() * 2 - 1) * bound;
            }
            coef.push(row);
        }
        const intercept: number[] = new Array(fanOut);
        for (let j = 0; j < fanOut; j++) {
            intercept[j] = (rand() * 2 - 1) * bound;
        }
        coefs.push(coef);
        intercepts.push(intercept);
    }
    return { coefs, intercepts };
}

function copyWeights(w: MLPWeights): MLPWeights {
    return {
        coefs: w.coefs.map(coef => coef.map(row => row.slice())),
        intercepts: w.intercepts.map(row => row.slice()),
    };
}

// ---------------------------------------------------------------------------
// Activations & losses
// ---------------------------------------------------------------------------

function applyHiddenActivation(Z: number[][], activation: MLPActivation): void {
    if (activation === 'identity') return;
    for (const row of Z) {
        for (let j = 0; j < row.length; j++) {
            const z = row[j];
            if (activation === 'relu') row[j] = z > 0 ? z : 0;
            else if (activation === 'tanh') row[j] = Math.tanh(z);
            else row[j] = 1 / (1 + Math.exp(-z)); // logistic
        }
    }
}

function applyOutActivation(Z: number[][], outActivation: MLPOutActivation): void {
    if (outActivation === 'identity') return;
    if (outActivation === 'logistic') {
        for (const row of Z) {
            for (let j = 0; j < row.length; j++) {
                row[j] = 1 / (1 + Math.exp(-row[j]));
            }
        }
        return;
    }
    // softmax
    for (const row of Z) {
        let max = -Infinity;
        for (const z of row) if (z > max) max = z;
        let sum = 0;
        for (let j = 0; j < row.length; j++) {
            row[j] = Math.exp(row[j] - max);
            sum += row[j];
        }
        for (let j = 0; j < row.length; j++) {
            row[j] /= sum;
        }
    }
}

/**
 * In-place delta *= f'(z) expressed through the activation VALUES (sklearn's
 * inplace_*_derivative), so no pre-activation cache is needed.
 */
function applyActivationDerivative(delta: number[][], act: number[][], activation: MLPActivation): void {
    if (activation === 'identity') return;
    for (let i = 0; i < delta.length; i++) {
        const d = delta[i];
        const a = act[i];
        for (let j = 0; j < d.length; j++) {
            if (activation === 'relu') {
                if (a[j] === 0) d[j] = 0;
            } else if (activation === 'tanh') {
                d[j] *= 1 - a[j] * a[j];
            } else { // logistic
                d[j] *= a[j] * (1 - a[j]);
            }
        }
    }
}

const LOSS_CLIP = 1e-10;

/** Mean data loss over the batch (sklearn's log_loss / binary_log_loss / squared_loss). */
function dataLoss(kind: MLPLossKind, Y: number[][], P: number[][]): number {
    const n = Y.length;
    let loss = 0;
    if (kind === 'squared') {
        let count = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < Y[i].length; j++) {
                const d = Y[i][j] - P[i][j];
                loss += d * d;
                count++;
            }
        }
        return loss / (2 * count); // mean over all entries, halved
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < Y[i].length; j++) {
            const p = Math.min(1 - LOSS_CLIP, Math.max(LOSS_CLIP, P[i][j]));
            if (kind === 'log') {
                if (Y[i][j] !== 0) loss -= Y[i][j] * Math.log(p);
            } else { // binaryLog
                loss -= Y[i][j] * Math.log(p) + (1 - Y[i][j]) * Math.log(1 - p);
            }
        }
    }
    return loss / n;
}

// ---------------------------------------------------------------------------
// Forward / backward
// ---------------------------------------------------------------------------

/**
 * Forward pass returning the activations of every layer
 * (activations[0] === X, activations[last] = network output).
 */
export function forwardMLP(
    X: number[][],
    weights: MLPWeights,
    activation: MLPActivation,
    outActivation: MLPOutActivation
): number[][][] {
    const activations: number[][][] = [X];
    const nLayers = weights.coefs.length;
    for (let l = 0; l < nLayers; l++) {
        const input = activations[l];
        const coef = weights.coefs[l];
        const intercept = weights.intercepts[l];
        const fanOut = intercept.length;
        const out: number[][] = new Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const row = new Array(fanOut);
            const x = input[i];
            for (let j = 0; j < fanOut; j++) {
                let sum = intercept[j];
                for (let k = 0; k < x.length; k++) {
                    sum += x[k] * coef[k][j];
                }
                row[j] = sum;
            }
            out[i] = row;
        }
        if (l === nLayers - 1) applyOutActivation(out, outActivation);
        else applyHiddenActivation(out, activation);
        activations.push(out);
    }
    return activations;
}

/** Network output only (last layer of `forwardMLP`). */
export function predictMLPOutput(
    X: number[][],
    weights: MLPWeights,
    activation: MLPActivation,
    outActivation: MLPOutActivation
): number[][] {
    const activations = forwardMLP(X, weights, activation, outActivation);
    return activations[activations.length - 1];
}

interface Gradients {
    coefGrads: number[][][];
    interceptGrads: number[][];
}

/**
 * Backprop over one minibatch (sklearn `_backprop`). Returns the regularized
 * batch loss and the gradients. The L2 penalty applies to weights only, not
 * biases: loss += 0.5 * alpha * sum(coef^2) / nBatch and
 * coefGrad = (act^T·delta + alpha·coef) / nBatch.
 */
function backpropBatch(
    Xb: number[][],
    Yb: number[][],
    weights: MLPWeights,
    activation: MLPActivation,
    outActivation: MLPOutActivation,
    lossKind: MLPLossKind,
    alpha: number
): { loss: number; grads: Gradients } {
    const nB = Xb.length;
    const nLayers = weights.coefs.length;
    const activations = forwardMLP(Xb, weights, activation, outActivation);
    const P = activations[nLayers];

    let loss = dataLoss(lossKind, Yb, P);
    let sumSq = 0;
    for (const coef of weights.coefs) {
        for (const row of coef) {
            for (const w of row) sumSq += w * w;
        }
    }
    loss += (0.5 * alpha * sumSq) / nB;

    const coefGrads: number[][][] = new Array(nLayers);
    const interceptGrads: number[][] = new Array(nLayers);

    // canonical-link output: delta = P - Y for softmax+CE, logistic+BCE,
    // identity+squared loss alike
    let delta: number[][] = P.map((row, i) => row.map((p, j) => p - Yb[i][j]));

    for (let l = nLayers - 1; l >= 0; l--) {
        const act = activations[l]; // input to layer l
        const coef = weights.coefs[l];
        const fanIn = coef.length;
        const fanOut = coef[0].length;

        const cg: number[][] = new Array(fanIn);
        for (let i = 0; i < fanIn; i++) {
            cg[i] = new Array(fanOut).fill(0);
        }
        const ig: number[] = new Array(fanOut).fill(0);
        for (let s = 0; s < nB; s++) {
            const a = act[s];
            const d = delta[s];
            for (let j = 0; j < fanOut; j++) {
                ig[j] += d[j];
                for (let i = 0; i < fanIn; i++) {
                    cg[i][j] += a[i] * d[j];
                }
            }
        }
        for (let i = 0; i < fanIn; i++) {
            for (let j = 0; j < fanOut; j++) {
                cg[i][j] = (cg[i][j] + alpha * coef[i][j]) / nB;
            }
        }
        for (let j = 0; j < fanOut; j++) {
            ig[j] /= nB;
        }
        coefGrads[l] = cg;
        interceptGrads[l] = ig;

        if (l > 0) {
            // delta_{l-1} = delta_l · coef_l^T ⊙ f'(act_l)
            const prev: number[][] = new Array(nB);
            for (let s = 0; s < nB; s++) {
                const d = delta[s];
                const row = new Array(fanIn).fill(0);
                for (let i = 0; i < fanIn; i++) {
                    let sum = 0;
                    for (let j = 0; j < fanOut; j++) {
                        sum += d[j] * coef[i][j];
                    }
                    row[i] = sum;
                }
                prev[s] = row;
            }
            applyActivationDerivative(prev, act, activation);
            delta = prev;
        }
    }
    return { loss, grads: { coefGrads, interceptGrads } };
}

// ---------------------------------------------------------------------------
// Optimizers (fit-local; never stored on the estimator)
// ---------------------------------------------------------------------------

interface StochasticOptimizer {
    updateParams(weights: MLPWeights, grads: Gradients): void;
    /** Called once per epoch with the total number of samples seen. */
    iterationEnds(timeStep: number): void;
    /** true → stop training; false → keep going (adaptive lr was reduced). */
    triggerStopping(): boolean;
}

/** sklearn SGDOptimizer: (Nesterov) momentum + lr schedules. */
class SGDOptimizer implements StochasticOptimizer {
    private learningRate: number;
    private coefVel: number[][][];
    private interceptVel: number[][];

    constructor(
        weights: MLPWeights,
        private readonly learningRateInit: number,
        private readonly schedule: MLPLearningRateSchedule,
        private readonly momentum: number,
        private readonly nesterov: boolean,
        private readonly powerT: number
    ) {
        this.learningRate = learningRateInit;
        this.coefVel = weights.coefs.map(c => c.map(row => row.map(() => 0)));
        this.interceptVel = weights.intercepts.map(row => row.map(() => 0));
    }

    public updateParams(weights: MLPWeights, grads: Gradients): void {
        const { coefs, intercepts } = weights;
        const lr = this.learningRate;
        const m = this.momentum;
        for (let l = 0; l < coefs.length; l++) {
            const coef = coefs[l];
            const vel = this.coefVel[l];
            const g = grads.coefGrads[l];
            for (let i = 0; i < coef.length; i++) {
                for (let j = 0; j < coef[i].length; j++) {
                    vel[i][j] = m * vel[i][j] - lr * g[i][j];
                    coef[i][j] += this.nesterov ? m * vel[i][j] - lr * g[i][j] : vel[i][j];
                }
            }
            const intercept = intercepts[l];
            const ivel = this.interceptVel[l];
            const ig = grads.interceptGrads[l];
            for (let j = 0; j < intercept.length; j++) {
                ivel[j] = m * ivel[j] - lr * ig[j];
                intercept[j] += this.nesterov ? m * ivel[j] - lr * ig[j] : ivel[j];
            }
        }
    }

    public iterationEnds(timeStep: number): void {
        if (this.schedule === 'invscaling') {
            this.learningRate = this.learningRateInit / Math.pow(timeStep + 1, this.powerT);
        }
    }

    public triggerStopping(): boolean {
        if (this.schedule !== 'adaptive') {
            return true;
        }
        if (this.learningRate <= 1e-6) {
            return true;
        }
        this.learningRate /= 5;
        return false;
    }
}

/** sklearn AdamOptimizer: bias-corrected first/second moments. */
class AdamOptimizer implements StochasticOptimizer {
    private t = 0;
    private coefM: number[][][];
    private coefV: number[][][];
    private interceptM: number[][];
    private interceptV: number[][];

    constructor(
        weights: MLPWeights,
        private readonly learningRateInit: number,
        private readonly beta1: number,
        private readonly beta2: number,
        private readonly epsilon: number
    ) {
        this.coefM = weights.coefs.map(c => c.map(row => row.map(() => 0)));
        this.coefV = weights.coefs.map(c => c.map(row => row.map(() => 0)));
        this.interceptM = weights.intercepts.map(row => row.map(() => 0));
        this.interceptV = weights.intercepts.map(row => row.map(() => 0));
    }

    public updateParams(weights: MLPWeights, grads: Gradients): void {
        this.t += 1;
        const { beta1, beta2, epsilon } = this;
        const lr = (this.learningRateInit * Math.sqrt(1 - Math.pow(beta2, this.t))) /
            (1 - Math.pow(beta1, this.t));
        for (let l = 0; l < weights.coefs.length; l++) {
            const coef = weights.coefs[l];
            const g = grads.coefGrads[l];
            const M = this.coefM[l];
            const V = this.coefV[l];
            for (let i = 0; i < coef.length; i++) {
                for (let j = 0; j < coef[i].length; j++) {
                    M[i][j] = beta1 * M[i][j] + (1 - beta1) * g[i][j];
                    V[i][j] = beta2 * V[i][j] + (1 - beta2) * g[i][j] * g[i][j];
                    coef[i][j] -= (lr * M[i][j]) / (Math.sqrt(V[i][j]) + epsilon);
                }
            }
            const intercept = weights.intercepts[l];
            const ig = grads.interceptGrads[l];
            const iM = this.interceptM[l];
            const iV = this.interceptV[l];
            for (let j = 0; j < intercept.length; j++) {
                iM[j] = beta1 * iM[j] + (1 - beta1) * ig[j];
                iV[j] = beta2 * iV[j] + (1 - beta2) * ig[j] * ig[j];
                intercept[j] -= (lr * iM[j]) / (Math.sqrt(iV[j]) + epsilon);
            }
        }
    }

    public iterationEnds(): void {
        // Adam's step size is fully determined by t; nothing per-epoch.
    }

    public triggerStopping(): boolean {
        return true;
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

export interface MLPTrainConfig {
    outActivation: MLPOutActivation;
    lossKind: MLPLossKind;
    /**
     * Validation score for early stopping (higher is better): accuracy for
     * the classifier, R² for the regressor.
     */
    validationScore: (weights: MLPWeights, XVal: number[][], YVal: number[][]) => number;
}

export interface MLPTrainResult {
    weights: MLPWeights;
    lossCurve: number[];
    nIter: number;
    /** Best training loss (null when earlyStopping is on). */
    bestLoss: number | null;
    /** Per-epoch validation scores (empty when earlyStopping is off). */
    validationScores: number[];
    bestValidationScore: number | null;
}

/**
 * Minibatch training loop shared by both MLPs — a faithful port of sklearn's
 * `_fit_stochastic`, except that the early-stopping split is a plain
 * deterministic TAIL split (last validationFraction of the rows) instead of
 * sklearn's shuffled/stratified `train_test_split`.
 *
 * `Y` is the already-encoded target matrix (one-hot / 0-1 column / raw
 * column), one row per sample of `X`.
 */
export function trainMLP(
    X: number[][],
    Y: number[][],
    params: MLPParams,
    config: MLPTrainConfig
): MLPTrainResult {
    const rand = createRandomGenerator(params.randomState);
    const nFeatures = X[0].length;
    const nOutputs = Y[0].length;
    const layerUnits = [nFeatures, ...params.hiddenLayerSizes, nOutputs];
    const weights = initMLPWeights(layerUnits, params.activation, rand);

    // early-stopping split: plain tail split (documented deviation)
    let XTrain = X;
    let YTrain = Y;
    let XVal: number[][] = [];
    let YVal: number[][] = [];
    if (params.earlyStopping) {
        const nVal = Math.max(1, Math.floor(X.length * params.validationFraction));
        if (nVal >= X.length) {
            throw new Error('validationFraction leaves no training samples');
        }
        const cut = X.length - nVal;
        XTrain = X.slice(0, cut);
        YTrain = Y.slice(0, cut);
        XVal = X.slice(cut);
        YVal = Y.slice(cut);
    }

    const n = XTrain.length;
    const batchSize = params.batchSize === 'auto'
        ? Math.min(200, n)
        : Math.min(params.batchSize, n);

    const optimizer: StochasticOptimizer = params.solver === 'sgd'
        ? new SGDOptimizer(weights, params.learningRateInit, params.learningRate,
            params.momentum, params.nesterovsMomentum, params.powerT)
        : new AdamOptimizer(weights, params.learningRateInit, params.beta1,
            params.beta2, params.epsilon);

    const indices = Array.from({ length: n }, (_, i) => i);
    const lossCurve: number[] = [];
    const validationScores: number[] = [];
    let bestLoss = Infinity;
    let bestValidationScore = -Infinity;
    let bestWeights: MLPWeights | null = null;
    let noImprovementCount = 0;
    let timeStep = 0;
    let nIter = 0;

    for (let it = 0; it < params.maxIter; it++) {
        if (params.shuffle) {
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(rand() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
        }
        let accumulatedLoss = 0;
        for (let start = 0; start < n; start += batchSize) {
            const end = Math.min(start + batchSize, n);
            const Xb: number[][] = new Array(end - start);
            const Yb: number[][] = new Array(end - start);
            for (let i = start; i < end; i++) {
                Xb[i - start] = XTrain[indices[i]];
                Yb[i - start] = YTrain[indices[i]];
            }
            const { loss, grads } = backpropBatch(
                Xb, Yb, weights, params.activation, config.outActivation,
                config.lossKind, params.alpha
            );
            accumulatedLoss += loss * (end - start);
            optimizer.updateParams(weights, grads);
        }
        nIter = it + 1;
        const loss = accumulatedLoss / n;
        lossCurve.push(loss);
        timeStep += n;

        if (params.earlyStopping) {
            const score = config.validationScore(weights, XVal, YVal);
            validationScores.push(score);
            if (score < bestValidationScore + params.tol) {
                noImprovementCount += 1;
            } else {
                noImprovementCount = 0;
            }
            if (score > bestValidationScore) {
                bestValidationScore = score;
                bestWeights = copyWeights(weights);
            }
        } else {
            if (loss > bestLoss - params.tol) {
                noImprovementCount += 1;
            } else {
                noImprovementCount = 0;
            }
            if (loss < bestLoss) {
                bestLoss = loss;
            }
        }

        optimizer.iterationEnds(timeStep);

        if (noImprovementCount > params.nIterNoChange) {
            if (optimizer.triggerStopping()) {
                break;
            }
            noImprovementCount = 0;
        }
    }

    if (params.earlyStopping && bestWeights !== null) {
        // restore the best weights seen on the validation set
        weights.coefs = bestWeights.coefs;
        weights.intercepts = bestWeights.intercepts;
    }

    return {
        weights,
        lossCurve,
        nIter,
        bestLoss: params.earlyStopping ? null : bestLoss,
        validationScores,
        bestValidationScore: params.earlyStopping ? bestValidationScore : null,
    };
}
