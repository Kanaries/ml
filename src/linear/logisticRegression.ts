import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';

function sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
}

export interface LogisticRegressionProps {
    learningRate?: number;
    maxIter?: number;
}

export class LogisticRegression extends ClassifierBase {
    private weights: number[];
    private bias: number;
    private learningRate: number;
    private maxIter: number;
    private classes: number[];
    private fitted: boolean;

    constructor(props: LogisticRegressionProps = {}) {
        super();
        const { learningRate = 0.1, maxIter = 1000 } = props;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.weights = [];
        this.bias = 0;
        this.classes = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { learningRate: this.learningRate, maxIter: this.maxIter };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (this.classes.length !== 2) {
            throw new Error(`LogisticRegression supports exactly 2 classes, got ${this.classes.length}`);
        }
        // internally train on {0, 1}; classes_ maps back to the user's labels
        const y01 = trainY.map(v => (v === this.classes[1] ? 1 : 0));
        const nFeatures = trainX[0].length;
        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;
        for (let iter = 0; iter < this.maxIter; iter++) {
            const gradW = new Array(nFeatures).fill(0);
            let gradB = 0;
            for (let i = 0; i < trainX.length; i++) {
                const x = trainX[i];
                const y = y01[i];
                let z = this.bias;
                for (let j = 0; j < nFeatures; j++) {
                    z += this.weights[j] * x[j];
                }
                const pred = sigmoid(z);
                const diff = pred - y;
                for (let j = 0; j < nFeatures; j++) {
                    gradW[j] += diff * x[j];
                }
                gradB += diff;
            }
            for (let j = 0; j < nFeatures; j++) {
                this.weights[j] -= this.learningRate * gradW[j] / trainX.length;
            }
            this.bias -= this.learningRate * gradB / trainX.length;
        }
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('LogisticRegression must be fitted before calling predict');
        }
        const results: number[] = [];
        for (const x of testX) {
            let z = this.bias;
            for (let j = 0; j < x.length; j++) {
                z += this.weights[j] * x[j];
            }
            const p = sigmoid(z);
            results.push(p >= 0.5 ? this.classes[1] : this.classes[0]);
        }
        return results;
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }
}
registerEstimator('LogisticRegression', LogisticRegression);
