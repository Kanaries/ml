import { ClassifierBase } from '../base';
import { LinearSVC, LinearSVCProps } from './linearSVC';

export interface SVCProps extends LinearSVCProps {
    kernel?: 'linear' | 'rbf';
    gamma?: number;
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
    private gamma: number;
    private trainX: number[][];

    constructor(props: SVCProps = {}) {
        super();
        const { kernel = 'rbf', gamma = 1, ...rest } = props;
        this.kernel = kernel;
        this.gamma = gamma;
        this.svc = new LinearSVC(rest);
        this.trainX = [];
    }

    private transform(X: number[][]): number[][] {
        return X.map(x => this.trainX.map(tx => rbf(x, tx, this.gamma)));
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.trainX = trainX;
        if (this.kernel === 'linear') {
            this.svc.fit(trainX, trainY);
        } else {
            const gram = this.transform(trainX);
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
