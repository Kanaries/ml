import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { forEachNonZeroInRow, matrixShape, type NumericMatrix } from '../data';
import { ensureClassPrior, validateMatrix, validateXY } from './utils';

export interface BernoulliNBProps {
    alpha?: number;
    binarize?: number | null;
    fitPrior?: boolean;
    classPrior?: number[] | null;
}

export class BernoulliNB extends ClassifierBase {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
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

    public getParams(): Params {
        return {
            alpha: this.alpha,
            binarize: this.binarize,
            fitPrior: this.fitPrior,
            classPrior: this.classPrior,
        };
    }

    private transformedValue(value: number): number {
        return this.binarize === null ? value : (value > this.binarize ? 1 : 0);
    }

    public fit(trainX: NumericMatrix, trainY: number[]): void {
        validateXY(trainX, trainY);
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        const nClasses = this.classes.length;
        const [nSamples, nFeatures] = matrixShape(trainX);
        const prior = ensureClassPrior(this.classPrior, nClasses, 'classPrior');

        this.classCount = new Array(nClasses).fill(0);
        this.featureCount = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));

        const classIndex = new Map<number, number>();
        this.classes.forEach((c, i) => classIndex.set(c, i));

        for (let i = 0; i < nSamples; i++) {
            const idx = classIndex.get(trainY[i])!;
            this.classCount[idx] += 1;
        }

        // A negative threshold makes implicit sparse zeros active. In the
        // binarize=null case raw feature values are retained, matching
        // sklearn's Y.T @ X feature counts.
        const implicitValue = this.transformedValue(0);
        if (implicitValue !== 0) {
            for (let c = 0; c < nClasses; c++) {
                this.featureCount[c].fill(this.classCount[c] * implicitValue);
            }
        }
        for (let i = 0; i < nSamples; i++) {
            const idx = classIndex.get(trainY[i])!;
            forEachNonZeroInRow(trainX, i, (j, value) => {
                this.featureCount[idx][j] += this.transformedValue(value) - implicitValue;
            });
        }

        // class log prior
        if (prior) {
            this.classLogPrior = prior.map(p => Math.log(p));
        } else if (this.fitPrior) {
            // sklearn never smooths the class prior with alpha
            const totalCount = this.classCount.reduce((a, b) => a + b, 0);
            this.classLogPrior = this.classCount.map(c => Math.log(c / totalCount));
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

    private jointLogLikelihood(X: NumericMatrix): number[][] {
        const [nSamples, nFeatures] = matrixShape(X);
        if (nFeatures !== this.featureLogProb[0].length) {
            throw new Error('input feature size does not match fitted model');
        }
        const nClasses = this.classes.length;
        const jll: number[][] = Array.from({ length: nSamples }, () => new Array(nClasses).fill(0));
        const implicitValue = this.transformedValue(0);
        const baselineTotals = this.classes.map((_, c) => {
            let total = this.classLogPrior[c];
            for (let j = 0; j < nFeatures; j++) {
                total += implicitValue * this.featureLogProb[c][j]
                    + (1 - implicitValue) * this.negLogProb[c][j];
            }
            return total;
        });
        for (let i = 0; i < nSamples; i++) {
            for (let c = 0; c < nClasses; c++) {
                let sum = baselineTotals[c];
                forEachNonZeroInRow(X, i, (j, value) => {
                    const delta = this.transformedValue(value) - implicitValue;
                    sum += delta * (this.featureLogProb[c][j] - this.negLogProb[c][j]);
                });
                jll[i][c] = sum;
            }
        }
        return jll;
    }

    public predict(testX: NumericMatrix): number[] {
        if (this.classes.length === 0) throw new Error('BernoulliNB must be fitted before calling predict');
        const [nSamples] = matrixShape(testX);
        if (nSamples === 0) return [];
        validateMatrix(testX);
        const jll = this.jointLogLikelihood(testX);
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
registerEstimator('BernoulliNB', BernoulliNB);
