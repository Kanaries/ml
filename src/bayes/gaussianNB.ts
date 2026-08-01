import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { forEachNonZeroInRow, matrixRow, matrixShape, type NumericMatrix } from '../data';
import { argmax, ensureClassPrior, sortedUniqueLabels, validateMatrix, validateXY } from './utils';

export interface GaussianNBProps {
    priors?: number[] | null;
    varSmoothing?: number;
}

export class GaussianNB extends ClassifierBase {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
    private priors: number[] | null;
    private varSmoothing: number;
    private classes: number[] = [];
    private theta: number[][] = [];
    private variances: number[][] = [];
    private classPrior: number[] = [];
    private epsilon = 1e-9;
    private fitted = false;

    constructor(props: GaussianNBProps = {}) {
        super();
        const { priors = null, varSmoothing = 1e-9 } = props;
        if (!Number.isFinite(varSmoothing) || varSmoothing < 0) {
            throw new Error('varSmoothing must be a finite number >= 0');
        }
        if (priors) {
            ensureClassPrior(priors, priors.length, 'priors');
        }
        this.priors = priors;
        this.varSmoothing = varSmoothing;
    }

    public getParams(): Params {
        return { priors: this.priors, varSmoothing: this.varSmoothing };
    }

    public fit(X: NumericMatrix, y: number[]): void {
        validateXY(X, y);

        const classes = sortedUniqueLabels(y);
        const nClasses = classes.length;
        const [nSamples, nFeatures] = matrixShape(X);
        const priors = ensureClassPrior(this.priors, nClasses, 'priors');

        const totalSum = new Array(nFeatures).fill(0);
        const totalNnz = new Array(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) forEachNonZeroInRow(X, i, (j, value) => {
            totalSum[j] += value;
            totalNnz[j] += 1;
        });
        const totalMean = totalSum.map(sum => sum / nSamples);
        const totalCentered = totalMean.map((mean, j) => (nSamples - totalNnz[j]) * mean * mean);
        for (let i = 0; i < nSamples; i++) forEachNonZeroInRow(X, i, (j, value) => {
            const diff = value - totalMean[j];
            totalCentered[j] += diff * diff;
        });
        const featureVar = totalCentered.map(sum => sum / nSamples);
        let maxFeatureVariance = 0;
        for (const variance of featureVar) {
            if (variance > maxFeatureVariance) maxFeatureVariance = variance;
        }
        this.epsilon = this.varSmoothing * maxFeatureVariance;

        this.theta = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        this.variances = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        const classCount = new Array(nClasses).fill(0);
        const classIndex = new Map<number, number>();
        classes.forEach((label, index) => classIndex.set(label, index));
        const sums = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        const nnzCounts = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));

        for (let i = 0; i < nSamples; i++) {
            const c = classIndex.get(y[i])!;
            classCount[c] += 1;
            forEachNonZeroInRow(X, i, (j, value) => {
                sums[c][j] += value;
                nnzCounts[c][j] += 1;
            });
        }
        for (let c = 0; c < nClasses; c++) {
            for (let j = 0; j < nFeatures; j++) {
                const mean = sums[c][j] / classCount[c];
                this.theta[c][j] = mean;
                this.variances[c][j] = (classCount[c] - nnzCounts[c][j]) * mean * mean;
            }
        }
        for (let i = 0; i < nSamples; i++) {
            const c = classIndex.get(y[i])!;
            forEachNonZeroInRow(X, i, (j, value) => {
                const diff = value - this.theta[c][j];
                this.variances[c][j] += diff * diff;
            });
        }
        for (let c = 0; c < nClasses; c++) for (let j = 0; j < nFeatures; j++) {
            this.variances[c][j] = this.variances[c][j] / classCount[c] + this.epsilon;
        }

        this.classPrior = priors ?? classCount.map(count => count / nSamples);
        this.classes = classes;
        this.fitted = true;
    }

    private jointLogLikelihood(X: NumericMatrix): number[][] {
        if (!this.fitted) {
            throw new Error('GaussianNB must be fitted before calling predict');
        }
        validateMatrix(X);
        const [nSamples, nFeatures] = matrixShape(X);
        if (nFeatures !== this.theta[0].length) {
            throw new Error('input feature size does not match fitted model');
        }
        return Array.from({ length: nSamples }, (_, i) => {
            const row = matrixRow(X, i);
            return this.classes.map((_, classIndex) => {
            let score = Math.log(this.classPrior[classIndex]);
            for (let j = 0; j < row.length; j++) {
                const variance = this.variances[classIndex][j];
                const diff = row[j] - this.theta[classIndex][j];
                score += -0.5 * Math.log(2 * Math.PI * variance);
                score += -0.5 * (diff * diff) / variance;
            }
            return score;
            });
        });
    }

    public predict(X: NumericMatrix): number[] {
        return this.jointLogLikelihood(X).map(scores => this.classes[argmax(scores)]);
    }

    /** Class posteriors, columns ordered by sorted `classes` (sklearn's predict_proba). */
    public predictProba(X: NumericMatrix): number[][] {
        return this.jointLogLikelihood(X).map(scores => {
            const maxScore = Math.max(...scores);
            const exp = scores.map(s => Math.exp(s - maxScore));
            const total = exp.reduce((acc, v) => acc + v, 0);
            return exp.map(v => v / total);
        });
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }
}
registerEstimator('GaussianNB', GaussianNB);
