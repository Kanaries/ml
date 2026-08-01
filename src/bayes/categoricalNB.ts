import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { forEachNonZeroInRow, matrixRow, matrixShape, type NumericMatrix } from '../data';
import { ensureClassPrior, validateMatrix, validateXY } from './utils';

export interface CategoricalNBProps {
    alpha?: number;
    forceAlpha?: boolean;
    fitPrior?: boolean;
    classPrior?: number[] | null;
    minCategories?: number | number[] | null;
}

export class CategoricalNB extends ClassifierBase {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
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

    public getParams(): Params {
        return {
            alpha: this.alpha,
            forceAlpha: this.forceAlpha,
            fitPrior: this.fitPrior,
            classPrior: this.classPrior,
            minCategories: this.minCategories,
        };
    }

    private initCounters(X: NumericMatrix): void {
        const [nSamples, nFeatures] = matrixShape(X);
        this.nCategories = new Array(nFeatures).fill(0);
        const maxValues = new Array(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) {
            forEachNonZeroInRow(X, i, (j, value) => {
                if (!Number.isInteger(value) || value < 0) {
                    throw new Error('CategoricalNB requires non-negative integer feature values');
                }
                if (value > maxValues[j]) maxValues[j] = value;
            });
        }
        for (let j = 0; j < nFeatures; j++) {
            let minCat = 0;
            if (this.minCategories === null) {
                minCat = 0;
            } else if (typeof this.minCategories === 'number') {
                minCat = this.minCategories;
            } else {
                minCat = this.minCategories[j];
            }
            this.nCategories[j] = Math.max(maxValues[j] + 1, minCat);
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

    public fit(trainX: NumericMatrix, trainY: number[]): void {
        validateXY(trainX, trainY);
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        const classIndex = new Map<number, number>();
        this.classes.forEach((c, i) => classIndex.set(c, i));
        this.initCounters(trainX);
        const [nSamples, nFeatures] = matrixShape(trainX);

        for (let i = 0; i < nSamples; i++) {
            const ci = classIndex.get(trainY[i])!;
            this.classCount[ci] += 1;
        }

        // Implicit sparse zeros are category 0 for every feature.
        for (let j = 0; j < nFeatures; j++) {
            for (let c = 0; c < this.classes.length; c++) this.categoryCount[j][c][0] = this.classCount[c];
        }
        for (let i = 0; i < nSamples; i++) {
            const ci = classIndex.get(trainY[i])!;
            forEachNonZeroInRow(trainX, i, (j, value) => {
                if (value >= this.nCategories[j]) return;
                this.categoryCount[j][ci][0] -= 1;
                this.categoryCount[j][ci][value] += 1;
            });
        }

        const nClasses = this.classes.length;
        const prior = ensureClassPrior(this.classPrior, nClasses, 'classPrior');
        if (prior) {
            this.classLogPrior = prior.map(p => Math.log(p));
        } else if (this.fitPrior) {
            // sklearn never smooths the class prior with alpha
            const totalCount = this.classCount.reduce((a, b) => a + b, 0);
            this.classLogPrior = this.classCount.map(c => Math.log(c / totalCount));
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

    public predict(testX: NumericMatrix): number[] {
        if (this.classes.length === 0) throw new Error('CategoricalNB must be fitted before calling predict');
        const [nSamples, nFeatures] = matrixShape(testX);
        validateMatrix(testX);
        if (nFeatures !== this.nCategories.length) {
            throw new Error('input feature size does not match fitted model');
        }
        const nClasses = this.classes.length;
        const preds: number[] = [];
        for (let i = 0; i < nSamples; i++) {
            const row = matrixRow(testX, i);
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
registerEstimator('CategoricalNB', CategoricalNB);
