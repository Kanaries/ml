import { ClassifierBase } from '../base';
import { DecisionTreeRegressor } from '../tree';
import { createRandomGenerator } from '../utils';

export interface GradientBoostingClassifierProps {
    nEstimators?: number;
    n_estimators?: number;
    learningRate?: number;
    learning_rate?: number;
    maxDepth?: number;
    max_depth?: number;
    minSamplesSplit?: number;
    min_samples_split?: number;
    subsample?: number;
    maxFeatures?: number | 'sqrt' | 'log2';
    max_features?: number | 'sqrt' | 'log2';
    randomState?: number;
    random_state?: number;
}

function sigmoid(v: number): number {
    return 1 / (1 + Math.exp(-v));
}

/**
 * Binary gradient boosting with log loss, following sklearn's
 * GradientBoostingClassifier: F_0 is the log-odds of the positive class,
 * each round fits a regression tree to the negative gradient y - p and
 * replaces each leaf value with the Newton step
 * sum(residuals) / sum(p * (1 - p)) over the samples the tree was fit on.
 */
export class GradientBoostingClassifier extends ClassifierBase {
    private nEstimators: number;
    private learningRate: number;
    private maxDepth: number;
    private minSamplesSplit: number;
    private subsample: number;
    private maxFeatures?: number | 'sqrt' | 'log2';
    private randomState?: number;
    private estimators: DecisionTreeRegressor[];
    private classes: number[];
    private initF: number;
    private fitted: boolean;

    constructor(props: GradientBoostingClassifierProps = {}) {
        super();
        this.nEstimators = props.nEstimators ?? props.n_estimators ?? 100;
        this.learningRate = props.learningRate ?? props.learning_rate ?? 0.1;
        this.maxDepth = props.maxDepth ?? props.max_depth ?? 3;
        this.minSamplesSplit = props.minSamplesSplit ?? props.min_samples_split ?? 2;
        this.subsample = props.subsample ?? 1.0;
        this.maxFeatures = props.maxFeatures ?? props.max_features;
        this.randomState = props.randomState ?? props.random_state;
        this.estimators = [];
        this.classes = [];
        this.initF = 0;
        this.fitted = false;
        if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
            throw new Error('nEstimators must be a positive integer');
        }
        if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
            throw new Error('learningRate must be a positive finite number');
        }
        if (!(this.subsample > 0 && this.subsample <= 1)) {
            throw new Error('subsample must be in (0, 1]');
        }
        if (!(Number.isInteger(this.maxDepth) || this.maxDepth === Infinity) || this.maxDepth < 1) {
            throw new Error('maxDepth must be an integer >= 1 (or Infinity)');
        }
        if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
            throw new Error('minSamplesSplit must be an integer >= 2');
        }
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        // validate before mutating state so a failed refit leaves a
        // previously fitted model intact
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length !== 2) {
            throw new Error('GradientBoostingClassifier currently supports only binary classification');
        }
        this.classes = classes;
        const n = trainX.length;
        const yBin = trainY.map(v => (v === this.classes[1] ? 1 : 0));
        const random = createRandomGenerator(this.randomState);

        const pPos = yBin.reduce((a, b) => a + b, 0) / n;
        this.initF = Math.log(pPos / (1 - pPos));
        this.estimators = [];
        const F = new Array(n).fill(this.initF);
        const subsampleSize = Math.max(1, Math.floor(this.subsample * n));

        for (let m = 0; m < this.nEstimators; m++) {
            const p = F.map(sigmoid);
            const residuals = yBin.map((y, i) => y - p[i]);

            let rows: number[];
            if (subsampleSize < n) {
                const indices = Array.from({ length: n }, (_, i) => i);
                for (let i = 0; i < subsampleSize; i++) {
                    const j = i + Math.floor(random() * (n - i));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                rows = indices.slice(0, subsampleSize);
            } else {
                rows = Array.from({ length: n }, (_, i) => i);
            }

            const tree = new DecisionTreeRegressor({
                max_depth: this.maxDepth,
                min_samples_split: this.minSamplesSplit,
                max_features: this.maxFeatures,
                randomState: Math.floor(random() * 1_000_000_000),
            });
            tree.fit(rows.map(i => trainX[i]), rows.map(i => residuals[i]));

            // Newton step per leaf over the in-bag samples:
            // value = sum(residual) / sum(p * (1 - p))
            const leafIds = tree.apply(rows.map(i => trainX[i]));
            const numerator: Map<number, number> = new Map();
            const denominator: Map<number, number> = new Map();
            for (let k = 0; k < rows.length; k++) {
                const i = rows[k];
                const leaf = leafIds[k];
                numerator.set(leaf, (numerator.get(leaf) || 0) + residuals[i]);
                denominator.set(leaf, (denominator.get(leaf) || 0) + p[i] * (1 - p[i]));
            }
            const leafValues: Map<number, number> = new Map();
            for (const [leaf, num] of numerator) {
                const den = denominator.get(leaf) || 0;
                leafValues.set(leaf, den < 1e-150 ? 0 : num / den);
            }
            tree.setLeafValues(leafValues);

            const update = tree.predict(trainX);
            for (let i = 0; i < n; i++) {
                F[i] += this.learningRate * update[i];
            }
            this.estimators.push(tree);
        }
        this.fitted = true;
    }

    private decisionFunction(testX: number[][]): number[] {
        const F = new Array(testX.length).fill(this.initF);
        for (const tree of this.estimators) {
            const pred = tree.predict(testX);
            for (let i = 0; i < F.length; i++) {
                F[i] += this.learningRate * pred[i];
            }
        }
        return F;
    }

    /**
     * Returns [P(classes[0]), P(classes[1])] per sample.
     */
    public predictProba(testX: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return this.decisionFunction(testX).map(f => {
            const p = sigmoid(f);
            return [1 - p, p];
        });
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return this.decisionFunction(testX).map(f => (f >= 0 ? this.classes[1] : this.classes[0]));
    }
}
