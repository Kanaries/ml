import { ClassifierBase } from '../base';

export interface LinearSVCProps {
    C?: number;
    maxIter?: number;
    learningRate?: number;
}

export class LinearSVC extends ClassifierBase {
    private C: number;
    private maxIter: number;
    private learningRate: number;
    private classes: number[];
    private weights: number[][];
    private biases: number[];

    constructor(props: LinearSVCProps = {}) {
        super();
        const { C = 1, maxIter = 100, learningRate = 0.01 } = props;
        this.C = C;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
        this.classes = [];
        this.weights = [];
        this.biases = [];
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.classes = Array.from(new Set(trainY));
        const nFeatures = trainX[0].length;
        this.weights = [];
        this.biases = [];
        for (let k = 0; k < this.classes.length; k++) {
            const cls = this.classes[k];
            let w = new Array(nFeatures).fill(0);
            let b = 0;
            const y = trainY.map(v => (v === cls ? 1 : -1));
            for (let iter = 0; iter < this.maxIter; iter++) {
                for (let i = 0; i < trainX.length; i++) {
                    const xi = trainX[i];
                    const yi = y[i];
                    let wx = 0;
                    for (let j = 0; j < nFeatures; j++) {
                        wx += w[j] * xi[j];
                    }
                    wx += b;
                    if (yi * wx < 1) {
                        for (let j = 0; j < nFeatures; j++) {
                            w[j] = w[j] - this.learningRate * (w[j] - this.C * yi * xi[j]);
                        }
                        b += this.learningRate * this.C * yi;
                    } else {
                        for (let j = 0; j < nFeatures; j++) {
                            w[j] = w[j] - this.learningRate * w[j];
                        }
                    }
                }
            }
            this.weights.push(w);
            this.biases.push(b);
        }
    }

    public predict(testX: number[][]): number[] {
        const results: number[] = [];
        for (const x of testX) {
            let bestIndex = 0;
            let bestScore = -Infinity;
            for (let k = 0; k < this.classes.length; k++) {
                const w = this.weights[k];
                const b = this.biases[k];
                let score = b;
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
