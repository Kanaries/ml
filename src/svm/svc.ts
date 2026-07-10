import { ClassifierBase } from '../base';
import { LinearSVC, LinearSVCProps } from './linearSVC';

export interface SVCProps extends LinearSVCProps {
    kernel?: 'linear' | 'rbf';
    /** 'scale' = 1/(n_features * Var(X)) like sklearn; 'auto' = 1/n_features */
    gamma?: number | 'scale' | 'auto';
}

function rbf(x1: number[], x2: number[], gamma: number): number {
    let sum = 0;
    for (let i = 0; i < x1.length; i++) {
        const d = x1[i] - x2[i];
        sum += d * d;
    }
    return Math.exp(-gamma * sum);
}

export class SVC extends ClassifierBase {
    private svc: LinearSVC;
    private kernel: 'linear' | 'rbf';
    private gamma: number | 'scale' | 'auto';
    private gammaValue: number;
    private trainX: number[][];

    constructor(props: SVCProps = {}) {
        super();
        const { kernel = 'rbf', gamma = 'scale', ...rest } = props;
        this.kernel = kernel;
        this.gamma = gamma;
        this.gammaValue = 1;
        this.svc = new LinearSVC(rest);
        this.trainX = [];
    }

    private resolveGamma(X: number[][]): number {
        if (typeof this.gamma === 'number') {
            return this.gamma;
        }
        const nFeatures = X[0].length;
        if (this.gamma === 'auto') {
            return 1 / nFeatures;
        }
        // 'scale': 1 / (n_features * Var(X)) over ALL entries, like sklearn
        let sum = 0;
        let count = 0;
        for (const row of X) for (const v of row) { sum += v; count++; }
        const mean = sum / count;
        let varSum = 0;
        for (const row of X) for (const v of row) varSum += (v - mean) ** 2;
        const variance = varSum / count;
        return variance > 0 ? 1 / (nFeatures * variance) : 1;
    }

    private transform(X: number[][]): number[][] {
        return X.map(x => this.trainX.map(tx => rbf(x, tx, this.gammaValue)));
    }

    public fit(trainX: number[][], trainY: number[]): void {
        // deep copy: predictions must not silently move when the caller
        // mutates the array it passed to fit
        this.trainX = trainX.map(row => row.slice());
        if (this.kernel === 'linear') {
            this.svc.fit(this.trainX, trainY);
        } else {
            this.gammaValue = this.resolveGamma(this.trainX);
            const gram = this.transform(this.trainX);
            this.svc.fit(gram, trainY);
        }
    }

    public predict(testX: number[][]): number[] {
        if (this.kernel === 'linear') {
            return this.svc.predict(testX);
        }
        const features = this.transform(testX);
        return this.svc.predict(features);
    }
}
