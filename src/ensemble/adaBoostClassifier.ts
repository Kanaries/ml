import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';

export interface AdaBoostClassifierProps {
    nEstimators?: number;
    n_estimators?: number;
    learningRate?: number;
    learning_rate?: number;
}

interface Stump {
    feature: number;
    threshold: number;
    polarity: 1 | -1;
}

/**
 * Multiclass decision stump: x <= threshold predicts leftClass, otherwise
 * rightClass. Classes are stored as indices into this.classes.
 */
interface MultiStump {
    feature: number;
    threshold: number;
    leftClass: number;
    rightClass: number;
}

function argmax(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

/**
 * AdaBoost classifier.
 *
 * Binary (K = 2): discrete AdaBoost over polarity stumps with the SAMME
 * one-sided weight update.
 *
 * Multiclass (K > 2): SAMME (Zhu et al. 2009), sklearn's AdaBoostClassifier
 * algorithm. The weak learner is a weighted multiclass stump (each side
 * predicts its weighted-majority class), a stump is accepted while its
 * weighted error stays below 1 - 1/K, and
 * alpha = learning_rate * (log((1-err)/err) + log(K-1)).
 */
export class AdaBoostClassifier extends ClassifierBase {
    private nEstimators: number;
    private learningRate: number;
    private estimators: Stump[];
    private estimatorWeights: number[];
    private multiEstimators: MultiStump[];
    private multiWeights: number[];
    private classes: number[];
    private nFeatures: number;

    constructor(props: AdaBoostClassifierProps = {}) {
        super();
        const nEstimators = props.nEstimators ?? props.n_estimators ?? 50;
        const learningRate = props.learningRate ?? props.learning_rate ?? 1.0;
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.estimators = [];
        this.estimatorWeights = [];
        this.multiEstimators = [];
        this.multiWeights = [];
        this.classes = [];
        this.nFeatures = 0;
    }

    public getParams(): Params {
        // canonical camelCase keys; the snake_case aliases remain accepted
        // by the constructor
        return {
            nEstimators: this.nEstimators,
            learningRate: this.learningRate,
        };
    }

    private validateInput(X: number[][], y: number[]): void {
        if (X.length === 0 || y.length === 0) {
            throw new Error('Input data cannot be empty');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples');
        }
        if (X[0].length === 0) {
            throw new Error('Features cannot be empty');
        }
    }

    private isFitted(): boolean {
        return this.estimators.length > 0 || this.multiEstimators.length > 0;
    }

    /**
     * trainY here is already mapped to {-1, +1}.
     */
    private trainStump(trainX: number[][], trainY: number[], weights: number[]): { stump: Stump; error: number } {
        const nSamples = trainX.length;
        const nFeatures = trainX[0].length;
        let bestError = Infinity;
        let bestStump: Stump = { feature: 0, threshold: 0, polarity: 1 };

        for (let f = 0; f < nFeatures; f++) {
            const values = trainX.map(r => r[f]);
            const unique = Array.from(new Set(values)).sort((a, b) => a - b);

            const thresholds: number[] = [];
            for (let i = 0; i < unique.length - 1; i++) {
                thresholds.push((unique[i] + unique[i + 1]) / 2);
            }

            if (unique.length > 0) {
                thresholds.push(unique[0] - 0.1);
                thresholds.push(unique[unique.length - 1] + 0.1);
            }

            for (const thresh of thresholds) {
                for (const polarity of [1, -1] as const) {
                    let err = 0;
                    for (let i = 0; i < nSamples; i++) {
                        const pred = polarity * (trainX[i][f] >= thresh ? 1 : -1);
                        if (pred !== trainY[i]) {
                            err += weights[i];
                        }
                    }
                    if (err < bestError) {
                        bestError = err;
                        bestStump = { feature: f, threshold: thresh, polarity };
                    }
                }
            }
        }
        return { stump: bestStump, error: bestError };
    }

    /**
     * Raw stump output in {-1, +1}.
     */
    private stumpScore(stump: Stump, X: number[][]): number[] {
        const { feature, threshold, polarity } = stump;
        return X.map(row => polarity * (row[feature] >= threshold ? 1 : -1));
    }

    /**
     * Weighted-misclassification-optimal multiclass stump: for every
     * midpoint threshold each side predicts its weighted-majority class.
     * yIdx holds class indices; weights are assumed normalized to sum 1.
     */
    private trainMultiStump(
        trainX: number[][],
        yIdx: number[],
        weights: number[],
        K: number
    ): { stump: MultiStump; error: number } {
        const n = trainX.length;
        const nFeatures = trainX[0].length;
        const totalByClass = new Array(K).fill(0);
        let totalW = 0;
        for (let i = 0; i < n; i++) {
            totalByClass[yIdx[i]] += weights[i];
            totalW += weights[i];
        }
        const majority = argmax(totalByClass);
        // fallback when no feature separates the samples: predict the
        // weighted majority on both sides
        let bestStump: MultiStump = { feature: 0, threshold: 0, leftClass: majority, rightClass: majority };
        let bestError = totalW - totalByClass[majority];

        for (let f = 0; f < nFeatures; f++) {
            const order = Array.from({ length: n }, (_, i) => i).sort(
                (a, b) => trainX[a][f] - trainX[b][f]
            );
            const leftW = new Array(K).fill(0);
            const rightW = [...totalByClass];
            for (let pos = 0; pos < n - 1; pos++) {
                const i = order[pos];
                leftW[yIdx[i]] += weights[i];
                rightW[yIdx[i]] -= weights[i];
                const vCur = trainX[order[pos]][f];
                const vNext = trainX[order[pos + 1]][f];
                if (vCur === vNext) continue;
                const leftClass = argmax(leftW);
                const rightClass = argmax(rightW);
                const err = totalW - leftW[leftClass] - rightW[rightClass];
                if (err < bestError) {
                    bestError = err;
                    // a/2 + b/2 avoids overflow; the guard handles rounding up
                    // to b, which would empty the right side
                    const mid = vCur / 2 + vNext / 2;
                    bestStump = {
                        feature: f,
                        threshold: mid === vNext ? vCur : mid,
                        leftClass,
                        rightClass,
                    };
                }
            }
        }
        return { stump: bestStump, error: bestError };
    }

    /**
     * Class-index predictions of a multiclass stump.
     */
    private multiStumpPredict(stump: MultiStump, X: number[][]): number[] {
        return X.map(row => (row[stump.feature] <= stump.threshold ? stump.leftClass : stump.rightClass));
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.validateInput(trainX, trainY);

        // validate before mutating state so a failed refit leaves a
        // previously fitted model intact
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length < 2) {
            throw new Error('AdaBoost needs at least 2 classes');
        }
        this.classes = classes;
        this.nFeatures = trainX[0].length;
        this.estimators = [];
        this.estimatorWeights = [];
        this.multiEstimators = [];
        this.multiWeights = [];
        if (this.classes.length === 2) {
            this.fitBinary(trainX, trainY);
        } else {
            this.fitMulticlass(trainX, trainY);
        }
    }

    private fitBinary(trainX: number[][], trainY: number[]): void {
        // internal labels: classes[0] -> -1, classes[1] -> +1
        const yPm = trainY.map(v => (v === this.classes[1] ? 1 : -1));

        const n = trainX.length;
        let sampleWeights = new Array(n).fill(1 / n);

        for (let i = 0; i < this.nEstimators; i++) {
            const { stump, error } = this.trainStump(trainX, yPm, sampleWeights);

            if (error >= 0.5) {
                if (this.estimators.length === 0) {
                    this.estimators.push(stump);
                    this.estimatorWeights.push(0);
                }
                break;
            }

            if (error === 0) {
                this.estimators.push(stump);
                this.estimatorWeights.push(1);
                break;
            }

            const alpha = this.learningRate * Math.log((1 - error) / Math.max(error, 1e-10));
            const pred = this.stumpScore(stump, trainX);

            // exact SAMME update: only misclassified samples are up-weighted
            // (no cap, matching the multiclass path); error is bounded away
            // from 0 and the weights are renormalized below
            for (let j = 0; j < n; j++) {
                if (pred[j] !== yPm[j]) {
                    sampleWeights[j] *= Math.exp(alpha);
                }
            }

            const sum = sampleWeights.reduce((a, b) => a + b, 0);
            if (sum > 0) {
                for (let j = 0; j < n; j++) {
                    sampleWeights[j] /= sum;
                }
            } else {
                sampleWeights.fill(1 / n);
            }

            this.estimators.push(stump);
            this.estimatorWeights.push(alpha);
        }
    }

    private fitMulticlass(trainX: number[][], trainY: number[]): void {
        const n = trainX.length;
        const K = this.classes.length;
        const classIndex: Map<number, number> = new Map(this.classes.map((c, k) => [c, k]));
        const yIdx = trainY.map(v => classIndex.get(v));
        let sampleWeights = new Array(n).fill(1 / n);

        for (let m = 0; m < this.nEstimators; m++) {
            const { stump, error } = this.trainMultiStump(trainX, yIdx, sampleWeights, K);

            // a stump no better than random guessing (error >= 1 - 1/K)
            // cannot get a positive alpha
            if (error >= 1 - 1 / K) {
                if (this.multiEstimators.length === 0) {
                    this.multiEstimators.push(stump);
                    this.multiWeights.push(0);
                }
                break;
            }

            if (error === 0) {
                this.multiEstimators.push(stump);
                this.multiWeights.push(1);
                break;
            }

            const alpha =
                this.learningRate *
                (Math.log((1 - error) / Math.max(error, 1e-10)) + Math.log(K - 1));
            const pred = this.multiStumpPredict(stump, trainX);

            // exact SAMME update (no cap): error is bounded away from 0 and
            // the weights are renormalized below, so exp(alpha) stays finite
            for (let j = 0; j < n; j++) {
                if (pred[j] !== yIdx[j]) {
                    sampleWeights[j] *= Math.exp(alpha);
                }
            }

            const sum = sampleWeights.reduce((a, b) => a + b, 0);
            if (sum > 0) {
                for (let j = 0; j < n; j++) {
                    sampleWeights[j] /= sum;
                }
            } else {
                sampleWeights.fill(1 / n);
            }

            this.multiEstimators.push(stump);
            this.multiWeights.push(alpha);
        }
    }

    private decisionScores(testX: number[][]): number[] {
        const scores = new Array(testX.length).fill(0);
        for (let i = 0; i < this.estimators.length; i++) {
            const pred = this.stumpScore(this.estimators[i], testX);
            const alpha = this.estimatorWeights[i];
            for (let j = 0; j < scores.length; j++) {
                scores[j] += alpha * pred[j];
            }
        }
        return scores;
    }

    /**
     * Per-class alpha-weighted vote totals, indexed like this.classes.
     */
    private voteScores(testX: number[][]): number[][] {
        const K = this.classes.length;
        const votes = testX.map(() => new Array(K).fill(0));
        for (let m = 0; m < this.multiEstimators.length; m++) {
            const pred = this.multiStumpPredict(this.multiEstimators[m], testX);
            const alpha = this.multiWeights[m];
            for (let j = 0; j < votes.length; j++) {
                votes[j][pred[j]] += alpha;
            }
        }
        return votes;
    }

    public predict(testX: number[][]): number[] {
        if (!this.isFitted()) {
            throw new Error('Model must be fitted before making predictions');
        }
        if (this.classes.length === 2) {
            return this.decisionScores(testX).map(s => (s >= 0 ? this.classes[1] : this.classes[0]));
        }
        return this.voteScores(testX).map(votes => this.classes[argmax(votes)]);
    }

    /**
     * Returns per-sample probabilities ordered by the sorted class labels.
     * Binary uses a sigmoid over the ensemble score, multiclass the
     * normalized alpha-weighted vote share - both heuristic calibrations,
     * not sklearn's exact probability transform.
     */
    public predictProba(testX: number[][]): number[][] {
        if (!this.isFitted()) {
            throw new Error('Model must be fitted before making predictions');
        }
        if (this.classes.length === 2) {
            return this.decisionScores(testX).map(s => {
                const prob1 = 1 / (1 + Math.exp(-s));
                return [1 - prob1, prob1];
            });
        }
        const totalAlpha = this.multiWeights.reduce((a, b) => a + b, 0);
        const K = this.classes.length;
        return this.voteScores(testX).map(votes => {
            if (totalAlpha <= 0) {
                return new Array(K).fill(1 / K);
            }
            return votes.map(v => v / totalAlpha);
        });
    }

    public getFeatureImportances(): number[] {
        if (!this.isFitted()) {
            throw new Error('Model must be fitted before getting feature importances');
        }
        const features = this.classes.length === 2
            ? this.estimators.map(s => s.feature)
            : this.multiEstimators.map(s => s.feature);
        const weights = this.classes.length === 2 ? this.estimatorWeights : this.multiWeights;

        const importances = new Array(this.nFeatures).fill(0);
        const totalWeight = weights.reduce((a, b) => a + Math.abs(b), 0);
        if (totalWeight > 0) {
            for (let i = 0; i < features.length; i++) {
                importances[features[i]] += Math.abs(weights[i]) / totalWeight;
            }
        }
        return importances;
    }
}
registerEstimator('AdaBoostClassifier', AdaBoostClassifier);
