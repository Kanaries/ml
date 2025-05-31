import { ClassifierBase } from '../base';

export interface BernoulliNBProps {
    alpha?: number;
    binarize?: number | null;
    fitPrior?: boolean;
    classPrior?: number[] | null;
}

export class BernoulliNB extends ClassifierBase {
    private alpha: number;
    private binarize: number | null;
    private fitPrior: boolean;
    private classPrior: number[] | null;

    private classes: number[] = [];
    private classCount: number[] = [];
    private featureCount: number[][] = [];
    private classLogPrior: number[] = [];
    private featureLogProb: number[][] = [];
    private negLogProb: number[][] = [];

    constructor(props: BernoulliNBProps = {}) {
        super();
        const { alpha = 1.0, binarize = 0.0, fitPrior = true, classPrior = null } = props;
        this.alpha = alpha;
        this.binarize = binarize;
        this.fitPrior = fitPrior;
        this.classPrior = classPrior;
    }

    private binarizeX(X: number[][]): number[][] {
        if (this.binarize === null) return X.map(r => r.slice());
        const threshold = this.binarize;
        return X.map(row => row.map(v => (v > threshold ? 1 : 0)));
    }

    public fit(trainX: number[][], trainY: number[]): void {
        const X = this.binarizeX(trainX);
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        const nClasses = this.classes.length;
        const nFeatures = X[0].length;

        this.classCount = new Array(nClasses).fill(0);
        this.featureCount = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));

        const classIndex = new Map<number, number>();
        this.classes.forEach((c, i) => classIndex.set(c, i));

        for (let i = 0; i < X.length; i++) {
            const idx = classIndex.get(trainY[i])!;
            this.classCount[idx] += 1;
            for (let j = 0; j < nFeatures; j++) {
                this.featureCount[idx][j] += X[i][j];
            }
        }

        // class log prior
        if (this.classPrior) {
            this.classLogPrior = this.classPrior.map(p => Math.log(p));
        } else if (this.fitPrior) {
            const totalCount = this.classCount.reduce((a, b) => a + b, 0);
            this.classLogPrior = this.classCount.map(c => Math.log((c + this.alpha) / (totalCount + nClasses * this.alpha)));
        } else {
            this.classLogPrior = new Array(nClasses).fill(Math.log(1 / nClasses));
        }

        // feature log prob
        this.featureLogProb = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        this.negLogProb = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        for (let i = 0; i < nClasses; i++) {
            for (let j = 0; j < nFeatures; j++) {
                const fc = this.featureCount[i][j];
                const cc = this.classCount[i];
                const prob = (fc + this.alpha) / (cc + 2 * this.alpha);
                this.featureLogProb[i][j] = Math.log(prob);
                this.negLogProb[i][j] = Math.log(1 - prob);
            }
        }
    }

    private jointLogLikelihood(X: number[][]): number[][] {
        const nSamples = X.length;
        const nClasses = this.classes.length;
        const jll: number[][] = Array.from({ length: nSamples }, () => new Array(nClasses).fill(0));
        for (let i = 0; i < nSamples; i++) {
            for (let c = 0; c < nClasses; c++) {
                let sum = this.classLogPrior[c];
                for (let j = 0; j < X[i].length; j++) {
                    if (X[i][j]) {
                        sum += this.featureLogProb[c][j];
                    } else {
                        sum += this.negLogProb[c][j];
                    }
                }
                jll[i][c] = sum;
            }
        }
        return jll;
    }

    public predict(testX: number[][]): number[] {
        const X = this.binarizeX(testX);
        const jll = this.jointLogLikelihood(X);
        const preds: number[] = [];
        for (const row of jll) {
            let best = 0;
            let max = row[0];
            for (let i = 1; i < row.length; i++) {
                if (row[i] > max) {
                    max = row[i];
                    best = i;
                }
            }
            preds.push(this.classes[best]);
        }
        return preds;
    }
}
