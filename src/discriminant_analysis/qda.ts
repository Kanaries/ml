import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { jacobiSVD } from './linalg';
import {
    argmaxRow,
    resolvePriors,
    softmaxRows,
    sortedUniqueLabels,
    validatePredictInput,
    validateXy,
} from './common';

export interface QuadraticDiscriminantAnalysisProps {
    /** Class priors (in sorted-class order). Defaults to class frequencies. */
    priors?: number[];
    /** Regularization: per-class covariance eigenvalues become (1 - regParam) * s2 + regParam. */
    regParam?: number;
    /** Threshold on covariance eigenvalues for rank estimation. */
    tol?: number;
}

/**
 * Quadratic Discriminant Analysis matching sklearn's
 * `QuadraticDiscriminantAnalysis`: one Gaussian per class with its own
 * covariance, fitted through a per-class SVD of the centered class data
 * (never forming the covariance matrix), giving a quadratic decision
 * boundary log P(x|k) + log P(k).
 *
 * Deviation from sklearn: on rank-deficient class data with regParam = 0,
 * sklearn only warns about collinearity and then produces inf/NaN scores;
 * here directions whose covariance eigenvalue is <= tol are dropped from the
 * Mahalanobis distance and log-determinant, so results stay finite.
 */
export class QuadraticDiscriminantAnalysis extends ClassifierBase {
    private priors?: number[];
    private regParam: number;
    private tol: number;

    private classes: number[];
    private classPriors: number[];
    private means: number[][];
    /** per class: kept covariance eigenvalues (descending) */
    private scalings: number[][];
    /** per class: kept principal directions, rotations[k][c] is a length-nFeatures vector */
    private rotations: number[][][];
    private nFeatures: number;
    private fitted: boolean;

    constructor(props: QuadraticDiscriminantAnalysisProps = {}) {
        super();
        const { priors, regParam = 0, tol = 1e-4 } = props;
        this.priors = priors == null ? undefined : priors.slice();
        this.regParam = regParam;
        this.tol = tol;

        this.classes = [];
        this.classPriors = [];
        this.means = [];
        this.scalings = [];
        this.rotations = [];
        this.nFeatures = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            priors: this.priors == null ? undefined : this.priors.slice(),
            regParam: this.regParam,
            tol: this.tol,
        };
    }

    public fit(X: number[][], y: number[]): void {
        validateXy(X, y, 'QuadraticDiscriminantAnalysis');
        if (!Number.isFinite(this.regParam) || this.regParam < 0 || this.regParam > 1) {
            throw new Error('QuadraticDiscriminantAnalysis: regParam must be between 0 and 1');
        }
        const n = X.length;
        const p = X[0].length;
        const classes = sortedUniqueLabels(y);
        const K = classes.length;
        if (K < 2) {
            throw new Error(`QuadraticDiscriminantAnalysis: y must contain at least 2 classes, got ${K}`);
        }

        const labelToIdx = new Map<number, number>();
        classes.forEach((c, i) => labelToIdx.set(c, i));
        const counts = new Array<number>(K).fill(0);
        for (const v of y) counts[labelToIdx.get(v)!] += 1;
        for (let k = 0; k < K; k++) {
            if (counts[k] < 2) {
                throw new Error(
                    `QuadraticDiscriminantAnalysis: class ${classes[k]} has only ${counts[k]} sample(s); ` +
                    'covariance is ill-defined with fewer than 2'
                );
            }
        }

        this.classes = classes;
        this.nFeatures = p;
        this.classPriors = resolvePriors(this.priors, counts, n, 'QuadraticDiscriminantAnalysis');

        const means: number[][] = [];
        const scalings: number[][] = [];
        const rotations: number[][][] = [];
        for (let k = 0; k < K; k++) {
            const rows: number[][] = [];
            for (let i = 0; i < n; i++) {
                if (labelToIdx.get(y[i]) === k) rows.push(X[i]);
            }
            const nk = rows.length;
            const mean = new Array<number>(p).fill(0);
            for (const row of rows) for (let j = 0; j < p; j++) mean[j] += row[j];
            for (let j = 0; j < p; j++) mean[j] /= nk;
            means.push(mean);

            const centered = rows.map((row) => row.map((v, j) => v - mean[j]));
            const { S, Vt } = jacobiSVD(centered);
            // covariance eigenvalues with sklearn's regularization applied via singular values
            const s2 = S.map((s) => ((1 - this.regParam) * s * s) / (nk - 1) + this.regParam);
            const keptScalings: number[] = [];
            const keptRotations: number[][] = [];
            for (let r = 0; r < s2.length; r++) {
                if (s2[r] > this.tol) {
                    keptScalings.push(s2[r]);
                    keptRotations.push(Vt[r].slice());
                }
            }
            if (keptScalings.length === 0) {
                throw new Error(
                    `QuadraticDiscriminantAnalysis: class ${classes[k]} has zero variance in every direction; ` +
                    'increase regParam'
                );
            }
            scalings.push(keptScalings);
            rotations.push(keptRotations);
        }
        this.means = means;
        this.scalings = scalings;
        this.rotations = rotations;
        this.fitted = true;
    }

    private assertFitted(method: string): void {
        if (!this.fitted) {
            throw new Error(`QuadraticDiscriminantAnalysis must be fitted before calling ${method}`);
        }
    }

    /** Per-class log-posterior parts: -0.5 (Mahalanobis + logdet) + log prior. */
    private decisionScores(X: number[][]): number[][] {
        validatePredictInput(X, this.nFeatures, 'QuadraticDiscriminantAnalysis');
        const K = this.classes.length;
        const logDet: number[] = [];
        for (let k = 0; k < K; k++) {
            let ld = 0;
            for (const s2 of this.scalings[k]) ld += Math.log(s2);
            logDet.push(ld);
        }
        return X.map((row) => {
            const scores = new Array<number>(K).fill(0);
            for (let k = 0; k < K; k++) {
                const mean = this.means[k];
                let maha = 0;
                for (let c = 0; c < this.rotations[k].length; c++) {
                    const dir = this.rotations[k][c];
                    let proj = 0;
                    for (let j = 0; j < row.length; j++) proj += (row[j] - mean[j]) * dir[j];
                    maha += (proj * proj) / this.scalings[k][c];
                }
                scores[k] = -0.5 * (maha + logDet[k]) + Math.log(this.classPriors[k]);
            }
            return scores;
        });
    }

    public predict(testX: number[][]): number[] {
        this.assertFitted('predict');
        return this.decisionScores(testX).map((row) => this.classes[argmaxRow(row)]);
    }

    /** Softmax over the per-class log posteriors (sklearn parity). */
    public predictProba(testX: number[][]): number[][] {
        this.assertFitted('predictProba');
        return softmaxRows(this.decisionScores(testX));
    }

    /**
     * Binary: 1-D signed scores for `classes[1]` (sklearn convention).
     * Multiclass: per-class log-posterior parts, shape [nSamples][nClasses].
     */
    public decisionFunction(testX: number[][]): number[] | number[][] {
        this.assertFitted('decisionFunction');
        const scores = this.decisionScores(testX);
        if (this.classes.length === 2) {
            return scores.map((row) => row[1] - row[0]);
        }
        return scores;
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    public getMeans(): number[][] {
        this.assertFitted('getMeans');
        return this.means.map((row) => row.slice());
    }

    public getPriors(): number[] {
        this.assertFitted('getPriors');
        return this.classPriors.slice();
    }

    /** Per-class covariance matrices reconstructed from the kept eigenpairs. */
    public getCovariance(): number[][][] {
        this.assertFitted('getCovariance');
        const p = this.nFeatures;
        return this.classes.map((_, k) => {
            const cov: number[][] = [];
            for (let a = 0; a < p; a++) cov.push(new Array<number>(p).fill(0));
            for (let c = 0; c < this.rotations[k].length; c++) {
                const dir = this.rotations[k][c];
                const s2 = this.scalings[k][c];
                for (let a = 0; a < p; a++) {
                    for (let b = 0; b < p; b++) {
                        cov[a][b] += s2 * dir[a] * dir[b];
                    }
                }
            }
            return cov;
        });
    }
}
registerEstimator('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis);
