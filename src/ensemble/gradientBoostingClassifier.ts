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

function softmaxRow(margins: number[]): number[] {
    // subtract the max for numerical stability
    const max = Math.max(...margins);
    const exps = margins.map(m => Math.exp(m - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

/**
 * Gradient boosting classifier following sklearn's GradientBoostingClassifier.
 *
 * Binary (K = 2): log loss. F_0 is the log-odds of the positive class, each
 * round fits one regression tree to the negative gradient y - p and replaces
 * each leaf value with the Newton step sum(residuals) / sum(p * (1 - p)).
 *
 * Multiclass (K > 2): multinomial deviance. F_0k = log(prior_k), each round
 * fits K trees (one per class) to y_k - softmax_k(F) using the shared
 * pre-round probabilities, with the Newton step
 * (K-1)/K * sum(residuals) / sum(p * (1 - p)) per leaf.
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
    private multiEstimators: DecisionTreeRegressor[][];
    private classes: number[];
    private initF: number;
    private initFMulti: number[];
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
        this.multiEstimators = [];
        this.classes = [];
        this.initF = 0;
        this.initFMulti = [];
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

    private sampleRows(n: number, size: number, random: () => number): number[] {
        if (size >= n) {
            return Array.from({ length: n }, (_, i) => i);
        }
        const indices = Array.from({ length: n }, (_, i) => i);
        for (let i = 0; i < size; i++) {
            const j = i + Math.floor(random() * (n - i));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, size);
    }

    private buildTree(random: () => number): DecisionTreeRegressor {
        return new DecisionTreeRegressor({
            max_depth: this.maxDepth,
            min_samples_split: this.minSamplesSplit,
            max_features: this.maxFeatures,
            randomState: Math.floor(random() * 1_000_000_000),
        });
    }

    /**
     * Fit one tree to the residuals over the in-bag rows and replace its
     * leaf values with the Newton step
     * factor * sum(residuals) / sum(p * (1 - p)).
     */
    private fitNewtonTree(
        trainX: number[][],
        rows: number[],
        residuals: number[],
        p: number[],
        factor: number,
        random: () => number
    ): DecisionTreeRegressor {
        const tree = this.buildTree(random);
        tree.fit(rows.map(i => trainX[i]), rows.map(i => residuals[i]));
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
            leafValues.set(leaf, den < 1e-150 ? 0 : (factor * num) / den);
        }
        tree.setLeafValues(leafValues);
        return tree;
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
        if (classes.length < 2) {
            throw new Error('GradientBoostingClassifier needs at least 2 classes');
        }
        this.classes = classes;
        if (this.classes.length === 2) {
            this.fitBinary(trainX, trainY);
        } else {
            this.fitMulticlass(trainX, trainY);
        }
        this.fitted = true;
    }

    private fitBinary(trainX: number[][], trainY: number[]): void {
        const n = trainX.length;
        const yBin = trainY.map(v => (v === this.classes[1] ? 1 : 0));
        const random = createRandomGenerator(this.randomState);

        const pPos = yBin.reduce((a, b) => a + b, 0) / n;
        this.initF = Math.log(pPos / (1 - pPos));
        this.estimators = [];
        this.multiEstimators = [];
        const F = new Array(n).fill(this.initF);
        const subsampleSize = Math.max(1, Math.floor(this.subsample * n));

        for (let m = 0; m < this.nEstimators; m++) {
            const p = F.map(sigmoid);
            const residuals = yBin.map((y, i) => y - p[i]);
            const rows = this.sampleRows(n, subsampleSize, random);
            const tree = this.fitNewtonTree(trainX, rows, residuals, p, 1, random);
            const update = tree.predict(trainX);
            for (let i = 0; i < n; i++) {
                F[i] += this.learningRate * update[i];
            }
            this.estimators.push(tree);
        }
    }

    private fitMulticlass(trainX: number[][], trainY: number[]): void {
        const n = trainX.length;
        const K = this.classes.length;
        const classIndex: Map<number, number> = new Map(this.classes.map((c, k) => [c, k]));
        const yIdx = trainY.map(v => classIndex.get(v));
        const random = createRandomGenerator(this.randomState);

        this.initFMulti = this.classes.map((_, k) => {
            const prior = yIdx.filter(y => y === k).length / n;
            return Math.log(prior);
        });
        this.estimators = [];
        this.multiEstimators = [];
        // F[i][k]: margin of class k for sample i
        const F = Array.from({ length: n }, () => [...this.initFMulti]);
        const subsampleSize = Math.max(1, Math.floor(this.subsample * n));
        const factor = (K - 1) / K;

        for (let m = 0; m < this.nEstimators; m++) {
            // shared pre-round probabilities for all K trees of this round
            const P = F.map(softmaxRow);
            const rows = this.sampleRows(n, subsampleSize, random);
            const roundTrees: DecisionTreeRegressor[] = [];
            for (let k = 0; k < K; k++) {
                const pk = P.map(row => row[k]);
                const residuals = yIdx.map((y, i) => (y === k ? 1 : 0) - pk[i]);
                const tree = this.fitNewtonTree(trainX, rows, residuals, pk, factor, random);
                const update = tree.predict(trainX);
                for (let i = 0; i < n; i++) {
                    F[i][k] += this.learningRate * update[i];
                }
                roundTrees.push(tree);
            }
            this.multiEstimators.push(roundTrees);
        }
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

    private decisionFunctionMulti(testX: number[][]): number[][] {
        const F = testX.map(() => [...this.initFMulti]);
        for (const roundTrees of this.multiEstimators) {
            for (let k = 0; k < roundTrees.length; k++) {
                const pred = roundTrees[k].predict(testX);
                for (let i = 0; i < F.length; i++) {
                    F[i][k] += this.learningRate * pred[i];
                }
            }
        }
        return F;
    }

    /**
     * Returns per-sample probabilities ordered by the sorted class labels.
     */
    public predictProba(testX: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        if (this.classes.length === 2) {
            return this.decisionFunction(testX).map(f => {
                const p = sigmoid(f);
                return [1 - p, p];
            });
        }
        return this.decisionFunctionMulti(testX).map(softmaxRow);
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        if (this.classes.length === 2) {
            return this.decisionFunction(testX).map(f => (f >= 0 ? this.classes[1] : this.classes[0]));
        }
        return this.decisionFunctionMulti(testX).map(margins => {
            let best = 0;
            for (let k = 1; k < margins.length; k++) {
                if (margins[k] > margins[best]) best = k;
            }
            return this.classes[best];
        });
    }
}
