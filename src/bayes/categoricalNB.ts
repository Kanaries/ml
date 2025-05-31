import { ClassifierBase } from '../base';

export interface CategoricalNBProps {
    alpha?: number;
    forceAlpha?: boolean;
    fitPrior?: boolean;
    classPrior?: number[] | null;
    minCategories?: number | number[] | null;
}

export class CategoricalNB extends ClassifierBase {
    private alpha: number;
    private forceAlpha: boolean;
    private fitPrior: boolean;
    private classPrior: number[] | null;
    private minCategories: number | number[] | null;

    private classes: number[] = [];
    private classCount: number[] = [];
    private categoryCount: number[][][] = [];
    private classLogPrior: number[] = [];
    private featureLogProb: number[][][] = [];
    private nCategories: number[] = [];

    constructor(props: CategoricalNBProps = {}) {
        super();
        const {
            alpha = 1.0,
            forceAlpha = true,
            fitPrior = true,
            classPrior = null,
            minCategories = null
        } = props;
        this.alpha = forceAlpha ? alpha : Math.max(alpha, 1e-10);
        this.forceAlpha = forceAlpha;
        this.fitPrior = fitPrior;
        this.classPrior = classPrior;
        this.minCategories = minCategories;
    }

    private initCounters(X: number[][]): void {
        const nFeatures = X[0].length;
        this.nCategories = new Array(nFeatures).fill(0);
        for (let j = 0; j < nFeatures; j++) {
            let maxVal = 0;
            for (let i = 0; i < X.length; i++) {
                if (X[i][j] > maxVal) maxVal = X[i][j];
            }
            let minCat = 0;
            if (this.minCategories === null) {
                minCat = 0;
            } else if (typeof this.minCategories === 'number') {
                minCat = this.minCategories;
            } else {
                minCat = this.minCategories[j];
            }
            this.nCategories[j] = Math.max(maxVal + 1, minCat);
        }
        const nClasses = this.classes.length;
        this.categoryCount = [];
        this.featureLogProb = [];
        for (let j = 0; j < nFeatures; j++) {
            const cats = this.nCategories[j];
            const mat = Array.from({ length: nClasses }, () => new Array(cats).fill(0));
            this.categoryCount.push(mat.map(row => row.slice()));
            this.featureLogProb.push(mat.map(row => row.slice()));
        }
        this.classCount = new Array(nClasses).fill(0);
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        const classIndex = new Map<number, number>();
        this.classes.forEach((c, i) => classIndex.set(c, i));
        this.initCounters(trainX);
        const nFeatures = trainX[0].length;

        for (let i = 0; i < trainX.length; i++) {
            const ci = classIndex.get(trainY[i])!;
            this.classCount[ci] += 1;
            for (let j = 0; j < nFeatures; j++) {
                const v = trainX[i][j];
                if (v >= this.nCategories[j]) continue;
                this.categoryCount[j][ci][v] += 1;
            }
        }

        const nClasses = this.classes.length;
        if (this.classPrior) {
            this.classLogPrior = this.classPrior.map(p => Math.log(p));
        } else if (this.fitPrior) {
            const totalCount = this.classCount.reduce((a, b) => a + b, 0);
            this.classLogPrior = this.classCount.map(c => Math.log((c + this.alpha) / (totalCount + nClasses * this.alpha)));
        } else {
            this.classLogPrior = new Array(nClasses).fill(Math.log(1 / nClasses));
        }

        for (let j = 0; j < nFeatures; j++) {
            for (let c = 0; c < nClasses; c++) {
                for (let k = 0; k < this.nCategories[j]; k++) {
                    const count = this.categoryCount[j][c][k];
                    const denom = this.classCount[c] + this.nCategories[j] * this.alpha;
                    this.featureLogProb[j][c][k] = Math.log((count + this.alpha) / denom);
                }
            }
        }
    }

    public predict(testX: number[][]): number[] {
        const nFeatures = testX[0].length;
        const nClasses = this.classes.length;
        const preds: number[] = [];
        for (const row of testX) {
            let bestIdx = 0;
            let bestScore = -Infinity;
            for (let c = 0; c < nClasses; c++) {
                let score = this.classLogPrior[c];
                for (let j = 0; j < nFeatures; j++) {
                    const v = row[j];
                    if (v < this.nCategories[j]) {
                        score += this.featureLogProb[j][c][v];
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = c;
                }
            }
            preds.push(this.classes[bestIdx]);
        }
        return preds;
    }
}
