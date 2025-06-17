import { DecisionTreeRegressor } from '../tree/decisionTreeRegressor';

function weightedSampleIndices(weights: number[], size: number): number[] {
    const cumulative: number[] = [];
    let sum = 0;
    for (const w of weights) {
        sum += w;
        cumulative.push(sum);
    }
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
        const r = Math.random() * sum;
        let idx = 0;
        while (idx < cumulative.length - 1 && r > cumulative[idx]) {
            idx++;
        }
        result.push(Math.min(idx, weights.length - 1));
    }
    return result;
}

export interface AdaBoostRegressorProps {
    estimator?: DecisionTreeRegressor;
    n_estimators?: number;
    learning_rate?: number;
}

export class AdaBoostRegressor {
    private estimator: DecisionTreeRegressor;
    private n_estimators: number;
    private learning_rate: number;
    private estimators: DecisionTreeRegressor[] = [];
    private estimator_weights: number[] = [];

    constructor(props: AdaBoostRegressorProps = {}) {
        const { estimator = new DecisionTreeRegressor({ max_depth: 3 }), n_estimators = 50, learning_rate = 1.0 } = props;
        this.estimator = estimator;
        this.n_estimators = n_estimators;
        this.learning_rate = learning_rate;
    }

    public fit(X: number[][], y: number[]): void {
        if (X.length === 0) return;
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }

        const n_samples = X.length;
        let sample_weights = new Array(n_samples).fill(1 / n_samples);
        this.estimators = [];
        this.estimator_weights = [];

        for (let m = 0; m < this.n_estimators; m++) {
            const indices = weightedSampleIndices(sample_weights, n_samples);
            const sampleX = indices.map(i => X[i]);
            const sampleY = indices.map(i => y[i]);
            const est = new DecisionTreeRegressor({
                max_depth: (this.estimator as any).max_depth ?? 3,
                min_samples_split:
                    (this.estimator as any).min_sample_split ??
                    (this.estimator as any).min_samples_split ??
                    2,
            });
            est.fit(sampleX, sampleY);
            const pred = est.predict(X);
            const errors = pred.map((p, i) => Math.abs(p - y[i]));
            const maxErr = Math.max(...errors);
            if (maxErr === 0) {
                continue;
            }
            const r = errors.map(e => e / maxErr);
            let weightedErr = 0;
            for (let i = 0; i < n_samples; i++) {
                weightedErr += sample_weights[i] * r[i];
            }
            if (weightedErr >= 0.5) {
                break;
            }
            const beta = weightedErr / (1 - weightedErr);
            for (let i = 0; i < n_samples; i++) {
                sample_weights[i] *= Math.pow(beta, (1 - r[i]));
            }
            const sumW = sample_weights.reduce((a, b) => a + b, 0);
            if (sumW === 0) {
                break;
            }
            sample_weights = sample_weights.map(w => w / sumW);
            const estWeight = this.learning_rate * Math.log(1 / beta);
            this.estimators.push(est);
            this.estimator_weights.push(estWeight);
        }
    }

    public predict(X: number[][]): number[] {
        if (this.estimators.length === 0) return new Array(X.length).fill(0);
        const result = new Array(X.length).fill(0);
        const weightSum = this.estimator_weights.reduce((a, b) => a + b, 0);
        for (let m = 0; m < this.estimators.length; m++) {
            const pred = this.estimators[m].predict(X);
            for (let i = 0; i < pred.length; i++) {
                result[i] += this.estimator_weights[m] * pred[i];
            }
        }
        return result.map(r => r / weightSum);
    }
}

