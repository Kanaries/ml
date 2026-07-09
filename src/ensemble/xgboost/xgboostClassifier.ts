import { XGBoostBase, XGBoostProps } from './xgboostBase';

export interface XGBoostClassifierProps extends XGBoostProps {}

function sigmoid(v: number): number {
    return 1 / (1 + Math.exp(-v));
}

/**
 * XGBoost with the binary:logistic objective: p = sigmoid(F),
 * g = p - y, h = p * (1 - p). The initial margin is
 * logit(base_score), 0 for the default base_score = 0.5.
 */
export class XGBoostClassifier extends XGBoostBase {
    private classes: number[];

    constructor(props: XGBoostClassifierProps = {}) {
        super(props);
        this.classes = [];
        if (!(this.baseScore > 0 && this.baseScore < 1)) {
            throw new Error('baseScore must be in (0, 1) for binary:logistic');
        }
    }

    protected gradients(F: number[], yBin: number[]): { g: number[]; h: number[] } {
        const p = F.map(sigmoid);
        return {
            g: p.map((pi, i) => pi - yBin[i]),
            h: p.map(pi => pi * (1 - pi)),
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.validateFitInput(trainX, trainY);
        // validate before mutating state so a failed refit leaves a
        // previously fitted model intact
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length !== 2) {
            throw new Error('XGBoostClassifier currently supports only binary classification');
        }
        this.classes = classes;
        const yBin = trainY.map(v => (v === this.classes[1] ? 1 : 0));
        const initialMargin = Math.log(this.baseScore / (1 - this.baseScore));
        this.fitBoosted(trainX, yBin, initialMargin);
    }

    /**
     * Returns [P(classes[0]), P(classes[1])] per sample.
     */
    public predictProba(testX: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return this.predictMargin(testX).map(f => {
            const p = sigmoid(f);
            return [1 - p, p];
        });
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return this.predictMargin(testX).map(f => (f >= 0 ? this.classes[1] : this.classes[0]));
    }
}
