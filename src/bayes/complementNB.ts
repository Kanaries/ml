import { argmax, classLogPriorFromCounts, ensureClassPrior, sortedUniqueLabels, validateXY } from './utils';

export interface ComplementNBProps {
    alpha?: number;
    forceAlpha?: boolean;
    fitPrior?: boolean;
    classPrior?: number[] | null;
    norm?: boolean;
}

export class ComplementNB {
    private alpha: number;
    private forceAlpha: boolean;
    private fitPrior: boolean;
    private classPrior: number[] | null;
    private norm: boolean;
    private classes: number[] = [];
    private classCount: number[] = [];
    private featureCount: number[][] = [];
    private featureAll: number[] = [];
    private classLogPrior: number[] = [];
    private featureLogProb: number[][] = [];
    private fitted = false;

    constructor(props: ComplementNBProps = {}) {
        const { alpha = 1.0, forceAlpha = true, fitPrior = true, classPrior = null, norm = false } = props;
        if (!Number.isFinite(alpha) || alpha < 0) {
            throw new Error('alpha must be a finite number >= 0');
        }
        if (classPrior) {
            ensureClassPrior(classPrior, classPrior.length, 'classPrior');
        }
        this.alpha = forceAlpha ? alpha : Math.max(alpha, 1e-10);
        this.forceAlpha = forceAlpha;
        this.fitPrior = fitPrior;
        this.classPrior = classPrior;
        this.norm = norm;
    }

    public fit(X: number[][], y: number[]): void {
        validateXY(X, y);
        const classes = sortedUniqueLabels(y);
        const nClasses = classes.length;
        const nFeatures = X[0].length;
        const prior = ensureClassPrior(this.classPrior, nClasses, 'classPrior');

        this.classes = classes;
        this.classCount = new Array(nClasses).fill(0);
        this.featureCount = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        this.featureAll = new Array(nFeatures).fill(0);

        const classIndex = new Map<number, number>();
        classes.forEach((label, index) => classIndex.set(label, index));

        for (let i = 0; i < X.length; i++) {
            const row = X[i];
            for (const value of row) {
                if (value < 0) {
                    throw new Error('ComplementNB requires non-negative feature values');
                }
            }
            const idx = classIndex.get(y[i])!;
            this.classCount[idx] += 1;
            for (let j = 0; j < nFeatures; j++) {
                this.featureCount[idx][j] += row[j];
                this.featureAll[j] += row[j];
            }
        }

        this.classLogPrior = classLogPriorFromCounts(this.classCount, this.fitPrior, prior);
        this.featureLogProb = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));

        for (let c = 0; c < nClasses; c++) {
            const compCount = this.featureAll.map((total, j) => total + this.alpha - this.featureCount[c][j]);
            const rowSum = compCount.reduce((acc, v) => acc + v, 0);
            const logged = compCount.map(v => Math.log(v / rowSum));
            const sumLogged = logged.reduce((acc, v) => acc + v, 0);
            for (let j = 0; j < nFeatures; j++) {
                this.featureLogProb[c][j] = this.norm ? logged[j] / sumLogged : -logged[j];
            }
        }

        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('ComplementNB must be fitted before calling predict');
        }
        validateXY(X, new Array(X.length).fill(0));
        if (X[0].length !== this.featureLogProb[0].length) {
            throw new Error('input feature size does not match fitted model');
        }
        return X.map(row => {
            const scores = this.classes.map((_, classIndex) => {
                let score = this.classLogPrior[classIndex];
                for (let j = 0; j < row.length; j++) {
                    score += row[j] * this.featureLogProb[classIndex][j];
                }
                return score;
            });
            return this.classes[argmax(scores)];
        });
    }
}
