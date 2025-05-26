import { ClassifierBase } from '../base';

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

    constructor(props: LogisticRegressionProps = {}) {
        super();
        const { learningRate = 0.1, maxIter = 1000 } = props;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.weights = [];
        this.bias = 0;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        const nFeatures = trainX[0].length;
        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;
        for (let iter = 0; iter < this.maxIter; iter++) {
            const gradW = new Array(nFeatures).fill(0);
            let gradB = 0;
            for (let i = 0; i < trainX.length; i++) {
                const x = trainX[i];
                const y = trainY[i];
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
    }

    public predict(testX: number[][]): number[] {
        const results: number[] = [];
        for (const x of testX) {
            let z = this.bias;
            for (let j = 0; j < x.length; j++) {
                z += this.weights[j] * x[j];
            }
            const p = sigmoid(z);
            results.push(p >= 0.5 ? 1 : 0);
        }
        return results;
    }
}
