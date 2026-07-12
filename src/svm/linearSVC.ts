import { ClassifierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';

export interface LinearSVCProps {
    C?: number;
    maxIter?: number;
    /** @deprecated ignored — Pegasos uses the schedule eta_t = 1/(lambda*t) */
    learningRate?: number;
    tol?: number;
    randomState?: number;
}

/**
 * Linear SVM trained with Pegasos (primal stochastic subgradient descent).
 *
 * Objective per class (one-vs-rest): lambda/2 ||w||^2 + (1/n) sum hinge,
 * with lambda = 1/(n*C) so the C semantics match sklearn's LinearSVC.
 * Step size follows the Pegasos schedule eta_t = 1/(lambda*t); the bias is
 * not regularized. maxIter counts epochs; training stops early when the
 * objective's relative improvement falls below tol.
 */
export class LinearSVC extends ClassifierBase {
    private C: number;
    private maxIter: number;
    /** @deprecated kept only so params round-trip; the Pegasos schedule ignores it */
    private learningRate?: number;
    private tol: number;
    private randomState?: number;
    private classes: number[];
    private weights: number[][];
    private biases: number[];

    constructor(props: LinearSVCProps = {}) {
        super();
        const { C = 1, maxIter = 1000, learningRate, tol = 1e-4, randomState } = props;
        if (!Number.isFinite(C) || C <= 0) {
            throw new Error('C must be a finite number > 0');
        }
        this.C = C;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
        this.tol = tol;
        this.randomState = randomState;
        this.classes = [];
        this.weights = [];
        this.biases = [];
    }

    public getParams(): Params {
        return {
            C: this.C,
            maxIter: this.maxIter,
            learningRate: this.learningRate,
            tol: this.tol,
            randomState: this.randomState,
        };
    }

    private objective(X: number[][], y: number[], w: number[], b: number, lambda: number): number {
        let normSq = 0;
        for (const v of w) normSq += v * v;
        let hinge = 0;
        for (let i = 0; i < X.length; i++) {
            let wx = b;
            for (let j = 0; j < w.length; j++) wx += w[j] * X[i][j];
            hinge += Math.max(0, 1 - y[i] * wx);
        }
        return (lambda / 2) * normSq + hinge / X.length;
    }

    private trainBinary(rawX: number[][], y: number[], rng: () => number): { w: number[]; b: number } {
        const n = rawX.length;
        const nFeatures = rawX[0].length;
        const lambda = 1 / (n * this.C);
        // center features: with an (effectively) unregularized intercept this
        // is an exact reparametrization of the same objective, and it moves
        // decision boundaries far from the origin to where SGD can reach them
        const mu = new Array(nFeatures).fill(0);
        for (const row of rawX) for (let j = 0; j < nFeatures; j++) mu[j] += row[j];
        for (let j = 0; j < nFeatures; j++) mu[j] /= n;
        const X = rawX.map(row => row.map((v, j) => v - mu[j]));
        // like liblinear, fold the intercept into w as a constant feature and
        // regularize it too — an unregularized bias gets near-cancelling +eta/-eta
        // updates. After centering the needed intercept is small, so its
        // penalty is negligible relative to the hinge term.
        const interceptScaling = 10;
        const dim = nFeatures + 1;
        const w = new Array(dim).fill(0);
        let t = 0;
        let prevObj = Infinity;
        const order = Array.from({ length: n }, (_, i) => i);
        for (let epoch = 0; epoch < this.maxIter; epoch++) {
            // shuffle each epoch (Fisher-Yates)
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [order[i], order[j]] = [order[j], order[i]];
            }
            for (const idx of order) {
                t++;
                const eta = 1 / (lambda * t);
                const xi = X[idx];
                const yi = y[idx];
                let wx = w[nFeatures] * interceptScaling;
                for (let j = 0; j < nFeatures; j++) wx += w[j] * xi[j];
                const scale = 1 - eta * lambda; // = 1 - 1/t
                for (let j = 0; j < dim; j++) w[j] *= scale;
                if (yi * wx < 1) {
                    // stochastic subgradient of lambda/2||w||^2 + E[hinge]:
                    // the sampled hinge estimates the mean, so no extra 1/n
                    const step = eta * yi;
                    for (let j = 0; j < nFeatures; j++) w[j] += step * xi[j];
                    w[nFeatures] += step * interceptScaling;
                }
            }
            const b = w[nFeatures] * interceptScaling;
            const obj = this.objective(X, y, w.slice(0, nFeatures), b, lambda);
            if (Math.abs(prevObj - obj) < this.tol * Math.max(1, Math.abs(prevObj))) {
                break;
            }
            prevObj = obj;
        }
        // un-center: f(x) = w.(x - mu) + b_c = w.x + (b_c - w.mu)
        const wOut = w.slice(0, nFeatures);
        let b = w[nFeatures] * interceptScaling;
        for (let j = 0; j < nFeatures; j++) b -= wOut[j] * mu[j];
        return { w: wOut, b };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (this.classes.length < 2) {
            throw new Error('LinearSVC requires at least 2 classes');
        }
        const rng = createRandomGenerator(this.randomState);
        this.weights = [];
        this.biases = [];
        if (this.classes.length === 2) {
            // one decision function is enough: the OvR twin is its negation
            const y = trainY.map(v => (v === this.classes[1] ? 1 : -1));
            const { w, b } = this.trainBinary(trainX, y, rng);
            this.weights = [w.map(v => -v), w];
            this.biases = [-b, b];
            return;
        }
        for (const cls of this.classes) {
            const y = trainY.map(v => (v === cls ? 1 : -1));
            const { w, b } = this.trainBinary(trainX, y, rng);
            this.weights.push(w);
            this.biases.push(b);
        }
    }

    public predict(testX: number[][]): number[] {
        if (this.weights.length === 0) {
            throw new Error('LinearSVC must be fitted before calling predict');
        }
        const results: number[] = [];
        for (const x of testX) {
            let bestIndex = 0;
            let bestScore = -Infinity;
            for (let k = 0; k < this.classes.length; k++) {
                const w = this.weights[k];
                let score = this.biases[k];
                for (let j = 0; j < x.length; j++) {
                    score += w[j] * x[j];
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = k;
                }
            }
            results.push(this.classes[bestIndex]);
        }
        return results;
    }
}
registerEstimator('LinearSVC', LinearSVC);
