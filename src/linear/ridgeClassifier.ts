import { ClassifierBase } from '../base';
import { RidgeRegression, RidgeRegressionProps } from './ridgeRegression';

function validateClassificationData(X: number[][], y: number[]): void {
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and y must be non-empty');
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
            throw new Error('all rows in X must have the same length');
        }
    }
}

export interface RidgeClassifierProps extends RidgeRegressionProps {}

export class RidgeClassifier extends ClassifierBase {
    private alpha: number;
    private fitIntercept: boolean;
    private classes: number[];
    private models: RidgeRegression[];

    constructor(props: RidgeClassifierProps = {}) {
        super();
        this.alpha = props.alpha ?? 1;
        this.fitIntercept = props.fitIntercept ?? true;
        this.classes = [];
        this.models = [];
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateClassificationData(trainX, trainY);
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        this.models = this.classes.map(targetClass => {
            const reg = new RidgeRegression({ alpha: this.alpha, fitIntercept: this.fitIntercept });
            const binaryTargets = trainY.map(label => (label === targetClass ? 1 : -1));
            reg.fit(trainX, binaryTargets);
            return reg;
        });
    }

    public predict(testX: number[][]): number[] {
        if (this.models.length === 0) {
            throw new Error('model is not fitted');
        }
        return testX.map(row => {
            let bestClass = this.classes[0];
            let bestScore = -Infinity;
            for (let i = 0; i < this.models.length; i++) {
                const score = this.models[i].predict([row])[0];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = this.classes[i];
                }
            }
            return bestClass;
        });
    }
}
