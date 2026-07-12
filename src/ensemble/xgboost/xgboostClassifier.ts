import { ClassifierBase } from '../../base';
import { registerEstimator, Params } from '../../base/estimator';
import { createRandomGenerator } from '../../utils';
import {
    XGBoostParams,
    XGBoostProps,
    fitBoostedTrees,
    predictBoostedMargin,
    resolveXGBoostProps,
    validateXGBoostFitInput,
} from './xgboostBase';
import { XGBTree } from './xgbTree';

export interface XGBoostClassifierProps extends XGBoostProps {}

function sigmoid(v: number): number {
    return 1 / (1 + Math.exp(-v));
}

function softmaxRow(margins: number[]): number[] {
    const max = Math.max(...margins);
    const exps = margins.map(m => Math.exp(m - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

/** binary:logistic objective: p = sigmoid(F), g = p - y, h = p * (1 - p). */
function binaryLogisticGradients(F: number[], yBin: number[]): { g: number[]; h: number[] } {
    const p = F.map(sigmoid);
    return {
        g: p.map((pi, i) => pi - yBin[i]),
        h: p.map(pi => pi * (1 - pi)),
    };
}

/**
 * XGBoost classifier.
 *
 * Binary (K = 2): binary:logistic. p = sigmoid(F), g = p - y,
 * h = p * (1 - p); the initial margin is logit(base_score).
 *
 * Multiclass (K > 2): multi:softprob. Each round builds K trees on the
 * shared pre-round softmax probabilities with g = p_k - 1{y=k} and
 * h = max(2 * p_k * (1 - p_k), 1e-16), matching the xgboost library's
 * softmax objective (multiclass_obj: kEps = 1e-16f). Initial margins are 0 (softmax is shift-invariant,
 * so base_score has no effect here).
 */
export class XGBoostClassifier extends ClassifierBase {
    private params: XGBoostParams;
    private trees: XGBTree[];
    private multiTrees: XGBTree[][];
    private baseMargin: number;
    private fitted: boolean;
    private classes: number[];

    constructor(props: XGBoostClassifierProps = {}) {
        super();
        this.params = resolveXGBoostProps(props);
        this.trees = [];
        this.multiTrees = [];
        this.baseMargin = 0;
        this.fitted = false;
        this.classes = [];
        if (!(this.params.baseScore > 0 && this.params.baseScore < 1)) {
            throw new Error('baseScore must be in (0, 1) for binary:logistic');
        }
    }

    public getParams(): Params {
        return { ...this.params };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateXGBoostFitInput(trainX, trainY);
        // validate before mutating state so a failed refit leaves a
        // previously fitted model intact
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length < 2) {
            throw new Error('XGBoostClassifier needs at least 2 classes');
        }
        this.classes = classes;
        if (this.classes.length === 2) {
            this.multiTrees = [];
            const yBin = trainY.map(v => (v === this.classes[1] ? 1 : 0));
            this.baseMargin = Math.log(this.params.baseScore / (1 - this.params.baseScore));
            this.trees = fitBoostedTrees(trainX, yBin, this.baseMargin, this.params, binaryLogisticGradients);
            this.fitted = true;
        } else {
            this.fitMulticlass(trainX, trainY);
        }
    }

    private fitMulticlass(trainX: number[][], trainY: number[]): void {
        const n = trainX.length;
        const K = this.classes.length;
        const nFeatures = trainX[0].length;
        const classIndex: Map<number, number> = new Map(this.classes.map((c, k) => [c, k]));
        const yIdx = trainY.map(v => classIndex.get(v));
        const random = createRandomGenerator(this.params.randomState);

        this.trees = [];
        this.multiTrees = [];
        this.baseMargin = 0;
        const F = Array.from({ length: n }, () => new Array(K).fill(0));
        const rowSampleSize = Math.max(1, Math.floor(this.params.subsample * n));
        const colSampleSize = Math.max(1, Math.floor(this.params.colsampleByTree * nFeatures));

        for (let m = 0; m < this.params.nEstimators; m++) {
            const P = F.map(softmaxRow);
            const roundTrees: XGBTree[] = [];
            for (let k = 0; k < K; k++) {
                const g = new Array(n);
                const h = new Array(n);
                for (let i = 0; i < n; i++) {
                    const p = P[i][k];
                    g[i] = p - (yIdx[i] === k ? 1 : 0);
                    h[i] = Math.max(2 * p * (1 - p), 1e-16);
                }

                let rowsX = trainX;
                let rowsG = g;
                let rowsH = h;
                if (rowSampleSize < n) {
                    const indices = Array.from({ length: n }, (_, i) => i);
                    for (let i = 0; i < rowSampleSize; i++) {
                        const j = i + Math.floor(random() * (n - i));
                        [indices[i], indices[j]] = [indices[j], indices[i]];
                    }
                    const rows = indices.slice(0, rowSampleSize);
                    rowsX = rows.map(i => trainX[i]);
                    rowsG = rows.map(i => g[i]);
                    rowsH = rows.map(i => h[i]);
                }

                let featureIndices = Array.from({ length: nFeatures }, (_, i) => i);
                if (colSampleSize < nFeatures) {
                    for (let i = 0; i < colSampleSize; i++) {
                        const j = i + Math.floor(random() * (nFeatures - i));
                        [featureIndices[i], featureIndices[j]] = [featureIndices[j], featureIndices[i]];
                    }
                    featureIndices = featureIndices.slice(0, colSampleSize);
                }

                const tree = new XGBTree({
                    maxDepth: this.params.maxDepth,
                    lambda: this.params.lambda,
                    gamma: this.params.gamma,
                    minChildWeight: this.params.minChildWeight,
                });
                tree.fit(rowsX, rowsG, rowsH, featureIndices);

                const update = tree.predict(trainX);
                for (let i = 0; i < n; i++) {
                    F[i][k] += this.params.learningRate * update[i];
                }
                roundTrees.push(tree);
            }
            this.multiTrees.push(roundTrees);
        }
        this.fitted = true;
    }

    private predictMarginBinary(testX: number[][]): number[] {
        return predictBoostedMargin(this.trees, this.baseMargin, this.params.learningRate, testX);
    }

    private predictMarginMulti(testX: number[][]): number[][] {
        const K = this.classes.length;
        const F = testX.map(() => new Array(K).fill(0));
        for (const roundTrees of this.multiTrees) {
            for (let k = 0; k < roundTrees.length; k++) {
                const pred = roundTrees[k].predict(testX);
                for (let i = 0; i < F.length; i++) {
                    F[i][k] += this.params.learningRate * pred[i];
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
            return this.predictMarginBinary(testX).map(f => {
                const p = sigmoid(f);
                return [1 - p, p];
            });
        }
        return this.predictMarginMulti(testX).map(softmaxRow);
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        if (this.classes.length === 2) {
            return this.predictMarginBinary(testX).map(f => (f >= 0 ? this.classes[1] : this.classes[0]));
        }
        return this.predictMarginMulti(testX).map(margins => {
            let best = 0;
            for (let k = 1; k < margins.length; k++) {
                if (margins[k] > margins[best]) best = k;
            }
            return this.classes[best];
        });
    }
}
registerEstimator('XGBoostClassifier', XGBoostClassifier);
