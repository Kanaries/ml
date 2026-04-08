import { argmax, ensureClassPrior, sortedUniqueLabels, validateMatrix, validateXY } from './utils';

export interface GaussianNBProps {
    priors?: number[] | null;
    varSmoothing?: number;
}

export class GaussianNB {
    private priors: number[] | null;
    private varSmoothing: number;
    private classes: number[] = [];
    private theta: number[][] = [];
    private variances: number[][] = [];
    private classPrior: number[] = [];
    private epsilon = 1e-9;
    private fitted = false;

    constructor(props: GaussianNBProps = {}) {
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

    public fit(X: number[][], y: number[]): void {
        validateXY(X, y);

        const classes = sortedUniqueLabels(y);
        const nClasses = classes.length;
        const nFeatures = X[0].length;
        const priors = ensureClassPrior(this.priors, nClasses, 'priors');

        const featureVar: number[] = new Array(nFeatures).fill(0);
        for (let j = 0; j < nFeatures; j++) {
            let mean = 0;
            for (let i = 0; i < X.length; i++) {
                mean += X[i][j];
            }
            mean /= X.length;
            let variance = 0;
            for (let i = 0; i < X.length; i++) {
                const diff = X[i][j] - mean;
                variance += diff * diff;
            }
            featureVar[j] = variance / X.length;
        }
        this.epsilon = this.varSmoothing * Math.max(...featureVar);

        this.theta = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        this.variances = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        const classCount = new Array(nClasses).fill(0);

        for (let c = 0; c < nClasses; c++) {
            const label = classes[c];
            const rows = X.filter((_, idx) => y[idx] === label);
            classCount[c] = rows.length;
            for (let j = 0; j < nFeatures; j++) {
                let mean = 0;
                for (const row of rows) {
                    mean += row[j];
                }
                mean /= rows.length;
                this.theta[c][j] = mean;

                let variance = 0;
                for (const row of rows) {
                    const diff = row[j] - mean;
                    variance += diff * diff;
                }
                this.variances[c][j] = variance / rows.length + this.epsilon;
            }
        }

        this.classPrior = priors ?? classCount.map(count => count / X.length);
        this.classes = classes;
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('GaussianNB must be fitted before calling predict');
        }
        validateMatrix(X);
        if (X[0].length !== this.theta[0].length) {
            throw new Error('input feature size does not match fitted model');
        }

        return X.map(row => {
            const scores = this.classes.map((_, classIndex) => {
                let score = Math.log(this.classPrior[classIndex]);
                for (let j = 0; j < row.length; j++) {
                    const variance = this.variances[classIndex][j];
                    const diff = row[j] - this.theta[classIndex][j];
                    score += -0.5 * Math.log(2 * Math.PI * variance);
                    score += -0.5 * (diff * diff) / variance;
                }
                return score;
            });
            return this.classes[argmax(scores)];
        });
    }
}
