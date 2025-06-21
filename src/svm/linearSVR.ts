export interface LinearSVRProps {
    epsilon?: number;
    C?: number;
    maxIter?: number;
    learningRate?: number;
}

export class LinearSVR {
    private epsilon: number;
    private C: number;
    private maxIter: number;
    private learningRate: number;
    private weights: number[];
    private bias: number;

    constructor(props: LinearSVRProps = {}) {
        const { epsilon = 0, C = 1, maxIter = 100, learningRate = 0.01 } = props;
        this.epsilon = epsilon;
        this.C = C;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
        this.weights = [];
        this.bias = 0;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        const nFeatures = trainX[0].length;
        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;
        for (let iter = 0; iter < this.maxIter; iter++) {
            for (let i = 0; i < trainX.length; i++) {
                const x = trainX[i];
                const y = trainY[i];
                let pred = this.bias;
                for (let j = 0; j < nFeatures; j++) {
                    pred += this.weights[j] * x[j];
                }
                const err = pred - y;
                if (Math.abs(err) > this.epsilon) {
                    const grad = err > 0 ? 1 : -1;
                    for (let j = 0; j < nFeatures; j++) {
                        this.weights[j] -= this.learningRate * (this.weights[j] + this.C * grad * x[j]);
                    }
                    this.bias -= this.learningRate * this.C * grad;
                } else {
                    for (let j = 0; j < nFeatures; j++) {
                        this.weights[j] -= this.learningRate * this.weights[j];
                    }
                }
            }
        }
    }

    public predict(testX: number[][]): number[] {
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
