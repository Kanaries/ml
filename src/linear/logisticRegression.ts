import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';

function sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
}

export interface LogisticRegressionProps {
    learningRate?: number;
    maxIter?: number;
    /** Inverse L2 regularization strength. `null` preserves the historical unregularized solver. */
    C?: number | null;
}

export class LogisticRegression extends ClassifierBase {
    private coefState: number[];
    private bias: number;
    private coefMatrixState: number[][];
    private biasState: number[];
    private learningRate: number;
    private maxIter: number;
    private C: number | null;
    private classes: number[];
    private fitted: boolean;

    constructor(props: LogisticRegressionProps = {}) {
        super();
        const { learningRate = 0.1, maxIter = 1000, C = null } = props;
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new Error('learningRate must be a finite number > 0');
        }
        if (!Number.isInteger(maxIter) || maxIter < 1) {
            throw new Error('maxIter must be a positive integer');
        }
        if (C !== null && (!Number.isFinite(C) || C <= 0)) {
            throw new Error('C must be null or a finite number > 0');
        }
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.C = C;
        this.coefState = [];
        this.bias = 0;
        this.coefMatrixState = [];
        this.biasState = [];
        this.classes = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { learningRate: this.learningRate, maxIter: this.maxIter, C: this.C ?? null };
    }

    private fitBinary(trainX: number[][], y01: number[]): { coef: number[]; bias: number } {
        const nFeatures = trainX[0].length;
        const coef = new Array(nFeatures).fill(0);
        let bias = 0;
        const l2 = this.C == null ? 0 : 1 / this.C;
        for (let iter = 0; iter < this.maxIter; iter++) {
            const gradW = new Array(nFeatures).fill(0);
            let gradB = 0;
            for (let i = 0; i < trainX.length; i++) {
                const x = trainX[i];
                let z = bias;
                for (let j = 0; j < nFeatures; j++) z += coef[j] * x[j];
                const diff = sigmoid(z) - y01[i];
                for (let j = 0; j < nFeatures; j++) gradW[j] += diff * x[j];
                gradB += diff;
            }
            for (let j = 0; j < nFeatures; j++) {
                // Implicit L2 step is stable even for very small C while
                // optimizing mean log-loss + ||w||²/(2*C*nSamples).
                coef[j] = (coef[j] - this.learningRate * gradW[j] / trainX.length)
                    / (1 + this.learningRate * l2 / trainX.length);
            }
            bias -= this.learningRate * gradB / trainX.length;
        }
        return { coef, bias };
    }

    private fitMulticlass(trainX: number[][], trainY: number[], classes: number[]): { coef: number[][]; bias: number[] } {
        const nFeatures = trainX[0].length, nClasses = classes.length;
        const coef = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        const bias = new Array(nClasses).fill(0);
        const classIndex = new Map(classes.map((label, index) => [label, index]));
        const l2 = this.C == null ? 0 : 1 / this.C;
        for (let iter = 0; iter < this.maxIter; iter++) {
            const gradW = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
            const gradB = new Array(nClasses).fill(0);
            for (let i = 0; i < trainX.length; i++) {
                const logits = coef.map((row, c) => row.reduce((sum, value, j) => sum + value * trainX[i][j], bias[c]));
                const maxLogit = Math.max(...logits);
                const exp = logits.map(value => Math.exp(value - maxLogit));
                const total = exp.reduce((sum, value) => sum + value, 0);
                const target = classIndex.get(trainY[i])!;
                for (let c = 0; c < nClasses; c++) {
                    const diff = exp[c] / total - Number(c === target);
                    for (let j = 0; j < nFeatures; j++) gradW[c][j] += diff * trainX[i][j];
                    gradB[c] += diff;
                }
            }
            for (let c = 0; c < nClasses; c++) {
                for (let j = 0; j < nFeatures; j++) {
                    coef[c][j] = (coef[c][j] - this.learningRate * gradW[c][j] / trainX.length)
                        / (1 + this.learningRate * l2 / trainX.length);
                }
                bias[c] -= this.learningRate * gradB[c] / trainX.length;
            }
        }
        return { coef, bias };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        // Defensive fit-time validation also protects deserialized payloads
        // and subclasses that did not pass through this constructor.
        if (!Number.isFinite(this.learningRate) || this.learningRate <= 0
            || !Number.isInteger(this.maxIter) || this.maxIter < 1
            || this.C != null && (!Number.isFinite(this.C) || this.C <= 0)) {
            throw new Error('invalid LogisticRegression parameters');
        }
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        if (trainX[0].length === 0 || trainX.some(row => row.length !== trainX[0].length || row.some(value => !Number.isFinite(value)))) {
            throw new Error('X must be a finite rectangular matrix with at least one feature');
        }
        if (trainY.some(value => !Number.isFinite(value))) {
            throw new Error('y must contain finite numeric labels');
        }
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length < 2) {
            throw new Error(`LogisticRegression requires at least 2 classes, got ${classes.length}`);
        }
        const models = classes.length === 2
            ? (() => { const model = this.fitBinary(trainX, trainY.map(value => value === classes[1] ? 1 : 0)); return { coef: [model.coef], bias: [model.bias] }; })()
            : this.fitMulticlass(trainX, trainY, classes);
        this.classes = classes;
        this.coefMatrixState = models.coef;
        this.biasState = models.bias;
        // Keep the historical binary fields populated for model compatibility.
        this.coefState = models.coef[0].slice();
        this.bias = models.bias[0];
        // Normalize models loaded from the pre-C serialization shape.
        this.C = this.C ?? null;
        this.fitted = true;
    }

    private scores(testX: number[][]): number[][] {
        if (!this.fitted) throw new Error('LogisticRegression must be fitted before calling predict');
        // Serialized pre-multiclass models only contain coefState/bias.
        const coefs = this.coefMatrixState?.length ? this.coefMatrixState : [this.coefState];
        const biases = this.biasState?.length ? this.biasState : [this.bias];
        return testX.map(x => coefs.map((coef, c) => {
            if (x.length !== coef.length || x.some(value => !Number.isFinite(value))) {
                throw new Error('input must be finite and match the fitted feature size');
            }
            return coef.reduce((sum, value, j) => sum + value * x[j], biases[c]);
        }));
    }

    public predict(testX: number[][]): number[] {
        return this.scores(testX).map(row => {
            if (this.classes.length === 2) return sigmoid(row[0]) >= 0.5 ? this.classes[1] : this.classes[0];
            let best = 0;
            for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
            return this.classes[best];
        });
    }

    public predictProba(testX: number[][]): number[][] {
        return this.scores(testX).map(row => {
            if (this.classes.length === 2) {
                const positive = sigmoid(row[0]);
                return [1 - positive, positive];
            }
            const maxScore = Math.max(...row);
            const probabilities = row.map(value => Math.exp(value - maxScore));
            const total = probabilities.reduce((sum, value) => sum + value, 0);
            return probabilities.map(value => value / total);
        });
    }

    public decisionFunction(testX: number[][]): number[] | number[][] {
        const result = this.scores(testX);
        return this.classes.length === 2 ? result.map(row => row[0]) : result;
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    public get coef(): number[] | number[][] {
        if (this.classes.length > 2) return this.coefMatrixState.map(row => row.slice());
        return this.coefState.slice();
    }
}
registerEstimator('LogisticRegression', LogisticRegression);
