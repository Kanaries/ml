import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { forEachNonZeroInRow, matrixShape, type NumericMatrix } from '../data';
import { argmax, classLogPriorFromCounts, ensureClassPrior, sortedUniqueLabels, validateMatrix, validateXY } from './utils';

export interface ComplementNBProps {
    alpha?: number;
    forceAlpha?: boolean;
    fitPrior?: boolean;
    classPrior?: number[] | null;
    norm?: boolean;
}

export class ComplementNB extends ClassifierBase {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
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
        super();
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

    public getParams(): Params {
        return {
            alpha: this.alpha,
            forceAlpha: this.forceAlpha,
            fitPrior: this.fitPrior,
            classPrior: this.classPrior,
            norm: this.norm,
        };
    }

    public fit(X: NumericMatrix, y: number[]): void {
        validateXY(X, y);
        const classes = sortedUniqueLabels(y);
        const nClasses = classes.length;
        const [nSamples, nFeatures] = matrixShape(X);
        const prior = ensureClassPrior(this.classPrior, nClasses, 'classPrior');

        this.classes = classes;
        this.classCount = new Array(nClasses).fill(0);
        this.featureCount = Array.from({ length: nClasses }, () => new Array(nFeatures).fill(0));
        this.featureAll = new Array(nFeatures).fill(0);

        const classIndex = new Map<number, number>();
        classes.forEach((label, index) => classIndex.set(label, index));

        for (let i = 0; i < nSamples; i++) {
            forEachNonZeroInRow(X, i, (_column, value) => {
                if (value < 0) {
                    throw new Error('ComplementNB requires non-negative feature values');
                }
            });
            const idx = classIndex.get(y[i])!;
            this.classCount[idx] += 1;
            forEachNonZeroInRow(X, i, (j, value) => {
                this.featureCount[idx][j] += value;
                this.featureAll[j] += value;
            });
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

    public predict(X: NumericMatrix): number[] {
        if (!this.fitted) {
            throw new Error('ComplementNB must be fitted before calling predict');
        }
        const [nSamples, nFeatures] = matrixShape(X);
        validateMatrix(X);
        if (nFeatures !== this.featureLogProb[0].length) {
            throw new Error('input feature size does not match fitted model');
        }
        // sklearn's ComplementNB JLL is X @ feature_log_prob.T with NO class
        // prior term (complement weights replace the prior to resist class
        // imbalance); the prior is only added in the degenerate single-class case.
        const singleClass = this.classes.length === 1;
        return Array.from({ length: nSamples }, (_, row) => {
            const scores = this.classes.map((_, classIndex) => {
                let score = singleClass ? this.classLogPrior[classIndex] : 0;
                forEachNonZeroInRow(X, row, (j, value) => {
                    score += value * this.featureLogProb[classIndex][j];
                });
                return score;
            });
            return this.classes[argmax(scores)];
        });
    }
}
registerEstimator('ComplementNB', ComplementNB);
