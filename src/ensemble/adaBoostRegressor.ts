import { DecisionTreeRegressor } from '../tree/decisionTreeRegressor';
import { createRandomGenerator } from '../utils';

function weightedSampleIndices(weights: number[], size: number, random: () => number): number[] {
    const cumulative: number[] = [];
    let sum = 0;
    for (const w of weights) {
        sum += w;
        cumulative.push(sum);
    }
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
        const r = random() * sum;
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
    nEstimators?: number;
    learning_rate?: number;
    learningRate?: number;
    randomState?: number;
    random_state?: number;
}

/**
 * AdaBoost.R2 (Drucker, 1997), matching sklearn's AdaBoostRegressor with
 * linear loss: weighted resampling per round, weighted-median prediction.
 */
export class AdaBoostRegressor {
    private estimator: DecisionTreeRegressor;
    private n_estimators: number;
    private learning_rate: number;
    private randomState?: number;
    private estimators: DecisionTreeRegressor[] = [];
    private estimator_weights: number[] = [];

    constructor(props: AdaBoostRegressorProps = {}) {
        const estimator = props.estimator ?? new DecisionTreeRegressor({ max_depth: 3 });
        const n_estimators = props.n_estimators ?? props.nEstimators ?? 50;
        const learning_rate = props.learning_rate ?? props.learningRate ?? 1.0;
        this.estimator = estimator;
        this.n_estimators = n_estimators;
        this.learning_rate = learning_rate;
        this.randomState = props.randomState ?? props.random_state;
    }

    public fit(X: number[][], y: number[]): void {
        if (X.length === 0) return;
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }

        const random = createRandomGenerator(this.randomState);
        const n_samples = X.length;
        let sample_weights = new Array(n_samples).fill(1 / n_samples);
        this.estimators = [];
        this.estimator_weights = [];

        for (let m = 0; m < this.n_estimators; m++) {
            const indices = weightedSampleIndices(sample_weights, n_samples, random);
            const sampleX = indices.map(i => X[i]);
            const sampleY = indices.map(i => y[i]);
            const est = new DecisionTreeRegressor({
                max_depth: (this.estimator as any).max_depth ?? 3,
                min_samples_split:
                    (this.estimator as any).min_sample_split ??
                    (this.estimator as any).min_samples_split ??
                    2,
                randomState: Math.floor(random() * 1_000_000_000),
            });
            est.fit(sampleX, sampleY);
            const pred = est.predict(X);
            const errors = pred.map((p, i) => Math.abs(p - y[i]));
            const maxErr = Math.max(...errors);
            if (maxErr === 0) {
                // perfect fit: keep the estimator and stop boosting
                this.estimators.push(est);
                this.estimator_weights.push(1);
                break;
            }
            const r = errors.map(e => e / maxErr);
            let weightedErr = 0;
            for (let i = 0; i < n_samples; i++) {
                weightedErr += sample_weights[i] * r[i];
            }
            if (weightedErr >= 0.5) {
                // never leave the ensemble empty (sklearn keeps the first estimator)
                if (this.estimators.length === 0) {
                    this.estimators.push(est);
                    this.estimator_weights.push(1);
                }
                break;
            }
            const beta = weightedErr / (1 - weightedErr);
            for (let i = 0; i < n_samples; i++) {
                sample_weights[i] *= Math.pow(beta, (1 - r[i]) * this.learning_rate);
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
        const allPreds = this.estimators.map(est => est.predict(X));
        const totalWeight = this.estimator_weights.reduce((a, b) => a + b, 0);
        return X.map((_, i) => {
            // weighted median of the estimators' predictions
            const pairs = this.estimators
                .map((_, m) => ({ pred: allPreds[m][i], weight: this.estimator_weights[m] }))
                .sort((a, b) => a.pred - b.pred);
            let cum = 0;
            for (const pair of pairs) {
                cum += pair.weight;
                if (cum >= totalWeight / 2) {
                    return pair.pred;
                }
            }
            return pairs[pairs.length - 1].pred;
        });
    }
}
