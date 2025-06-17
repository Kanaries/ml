import { ClassifierBase } from '../base';

export interface AdaBoostClassifierProps {
    nEstimators?: number;
    learningRate?: number;
    randomState?: number;
}

interface Stump {
    feature: number;
    threshold: number;
    polarity: 1 | -1;
}

export class AdaBoostClassifier extends ClassifierBase {
    private nEstimators: number;
    private learningRate: number;
    private estimators: Stump[];
    private estimatorWeights: number[];
    private classes: number[];

    constructor(props: AdaBoostClassifierProps = {}) {
        super();
        const { nEstimators = 50, learningRate = 1.0 } = props;
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.estimators = [];
        this.estimatorWeights = [];
        this.classes = [];
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
                        const label = trainY[i] === 1 ? 1 : -1;
                        if (pred !== label) {
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

    private stumpPredict(stump: Stump, X: number[][]): number[] {
        const { feature, threshold, polarity } = stump;
        return X.map(row => {
            const pred = polarity * (row[feature] >= threshold ? 1 : -1);
            return pred === 1 ? 1 : 0;
        });
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.validateInput(trainX, trainY);

        this.classes = Array.from(new Set(trainY)).sort();
        if (this.classes.length !== 2) {
            throw new Error('AdaBoost currently supports only binary classification');
        }

        const n = trainX.length;
        let sampleWeights = new Array(n).fill(1 / n);
        this.estimators = [];
        this.estimatorWeights = [];

        for (let i = 0; i < this.nEstimators; i++) {
            const { stump, error } = this.trainStump(trainX, trainY, sampleWeights);

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
            const pred = this.stumpPredict(stump, trainX);

            const maxWeight = Math.max(...sampleWeights) * 100;
            for (let j = 0; j < n; j++) {
                const sign = pred[j] === trainY[j] ? -1 : 1;
                sampleWeights[j] *= Math.exp(alpha * sign);
                sampleWeights[j] = Math.min(sampleWeights[j], maxWeight);
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

    public predict(testX: number[][]): number[] {
        if (this.estimators.length === 0) {
            throw new Error('Model must be fitted before making predictions');
        }

        const scores = new Array(testX.length).fill(0);
        for (let i = 0; i < this.estimators.length; i++) {
            const pred = this.stumpPredict(this.estimators[i], testX);
            const alpha = this.estimatorWeights[i];
            for (let j = 0; j < scores.length; j++) {
                scores[j] += alpha * (pred[j] === 1 ? 1 : -1);
            }
        }
        return scores.map(s => (s >= 0 ? 1 : 0));
    }

    public predictProba(testX: number[][]): number[][] {
        if (this.estimators.length === 0) {
            throw new Error('Model must be fitted before making predictions');
        }

        const scores = new Array(testX.length).fill(0);
        for (let i = 0; i < this.estimators.length; i++) {
            const pred = this.stumpPredict(this.estimators[i], testX);
            const alpha = this.estimatorWeights[i];
            for (let j = 0; j < scores.length; j++) {
                scores[j] += alpha * (pred[j] === 1 ? 1 : -1);
            }
        }

        return scores.map(s => {
            const prob1 = 1 / (1 + Math.exp(-s));
            return [1 - prob1, prob1];
        });
    }

    public getFeatureImportances(): number[] {
        if (this.estimators.length === 0) {
            throw new Error('Model must be fitted before getting feature importances');
        }

        const nFeatures = this.estimators.length > 0 ?
            Math.max(...this.estimators.map(s => s.feature)) + 1 : 0;
        const importances = new Array(nFeatures).fill(0);

        const totalWeight = this.estimatorWeights.reduce((a, b) => a + Math.abs(b), 0);
        if (totalWeight > 0) {
            for (let i = 0; i < this.estimators.length; i++) {
                const feature = this.estimators[i].feature;
                importances[feature] += Math.abs(this.estimatorWeights[i]) / totalWeight;
            }
        }

        return importances;
    }
}

