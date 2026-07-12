import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';

export interface LinearSVRProps {
    epsilon?: number;
    C?: number;
    maxIter?: number;
    /** @deprecated ignored — Pegasos uses the schedule eta_t = 1/(lambda*t) */
    learningRate?: number;
    tol?: number;
    randomState?: number;
}

/**
 * Linear SVR trained with Pegasos-style stochastic subgradient descent on
 * lambda/2 ||w||^2 + (1/n) sum max(0, |w.x + b - y| - epsilon), where
 * lambda = 1/(n*C) so the C semantics match sklearn's LinearSVR.
 */
export class LinearSVR extends RegressorBase {
    private epsilon: number;
    private C: number;
    private maxIter: number;
    /** @deprecated kept only so params round-trip; the Pegasos schedule ignores it */
    private learningRate?: number;
    private tol: number;
    private randomState?: number;
    private weights: number[];
    private bias: number;
    private fitted: boolean;

    constructor(props: LinearSVRProps = {}) {
        super();
        const { epsilon = 0, C = 1, maxIter = 1000, learningRate, tol = 1e-4, randomState } = props;
        if (!Number.isFinite(C) || C <= 0) {
            throw new Error('C must be a finite number > 0');
        }
        this.epsilon = epsilon;
        this.C = C;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
        this.tol = tol;
        this.randomState = randomState;
        this.weights = [];
        this.bias = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            epsilon: this.epsilon,
            C: this.C,
            maxIter: this.maxIter,
            learningRate: this.learningRate,
            tol: this.tol,
            randomState: this.randomState,
        };
    }

    private objective(X: number[][], Y: number[], lambda: number): number {
        let normSq = 0;
        for (const v of this.weights) normSq += v * v;
        let loss = 0;
        for (let i = 0; i < X.length; i++) {
            let pred = this.bias;
            for (let j = 0; j < this.weights.length; j++) pred += this.weights[j] * X[i][j];
            loss += Math.max(0, Math.abs(pred - Y[i]) - this.epsilon);
        }
        return (lambda / 2) * normSq + loss / X.length;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        const n = trainX.length;
        const nFeatures = trainX[0].length;
        const lambda = 1 / (n * this.C);
        // center X and y: exact reparametrization for the unregularized
        // intercept, and it keeps SGD stable when targets/features are large
        const mu = new Array(nFeatures).fill(0);
        for (const row of trainX) for (let j = 0; j < nFeatures; j++) mu[j] += row[j];
        for (let j = 0; j < nFeatures; j++) mu[j] /= n;
        let yMean = 0;
        for (const v of trainY) yMean += v;
        yMean /= n;
        const Xc = trainX.map(row => row.map((v, j) => v - mu[j]));
        const Yc = trainY.map(v => v - yMean);
        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;
        const rng = createRandomGenerator(this.randomState);
        const order = Array.from({ length: n }, (_, i) => i);
        let t = 0;
        let prevObj = Infinity;
        for (let epoch = 0; epoch < this.maxIter; epoch++) {
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [order[i], order[j]] = [order[j], order[i]];
            }
            for (const idx of order) {
                t++;
                const eta = 1 / (lambda * t);
                const x = Xc[idx];
                const y = Yc[idx];
                let pred = this.bias;
                for (let j = 0; j < nFeatures; j++) pred += this.weights[j] * x[j];
                const err = pred - y;
                const scale = 1 - eta * lambda; // = 1 - 1/t
                for (let j = 0; j < nFeatures; j++) this.weights[j] *= scale;
                if (Math.abs(err) > this.epsilon) {
                    const s = err > 0 ? 1 : -1;
                    const step = eta * s;
                    for (let j = 0; j < nFeatures; j++) this.weights[j] -= step * x[j];
                    this.bias -= step;
                }
            }
            const obj = this.objective(Xc, Yc, lambda);
            if (Math.abs(prevObj - obj) < this.tol * Math.max(1, Math.abs(prevObj))) {
                break;
            }
            prevObj = obj;
        }
        // un-center: f(x) = w.(x - mu) + b_c + yMean = w.x + (b_c + yMean - w.mu)
        this.bias += yMean;
        for (let j = 0; j < nFeatures; j++) this.bias -= this.weights[j] * mu[j];
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('LinearSVR must be fitted before calling predict');
        }
        const results: number[] = [];
        for (const x of testX) {
            let pred = this.bias;
            for (let j = 0; j < this.weights.length; j++) {
                pred += this.weights[j] * x[j];
            }
            results.push(pred);
        }
        return results;
    }
}
registerEstimator('LinearSVR', LinearSVR);
