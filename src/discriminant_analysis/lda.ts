import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { cholesky, jacobiSVD, solveLower, solveLowerTransposed, symmetricEigDecomposition } from './linalg';
import {
    argmaxRow,
    classMeans,
    resolvePriors,
    softmaxRows,
    sortedUniqueLabels,
    validatePredictInput,
    validateXy,
} from './common';

export type LDASolver = 'svd' | 'eigen';

export interface LinearDiscriminantAnalysisProps {
    /**
     * 'svd' (default, matches sklearn): no covariance matrix is formed, so it
     * is robust to many features and rank-deficient data. 'eigen' solves the
     * generalized eigenproblem Sw^-1 Sb and supports shrinkage.
     * sklearn's 'lsqr' solver is intentionally not implemented: it produces
     * the same classifier as 'eigen' but cannot transform.
     */
    solver?: LDASolver;
    /** Shrinkage strength in [0, 1] for the within-class covariance ('eigen' solver only). */
    shrinkage?: number;
    /** Class priors (in sorted-class order). Defaults to class frequencies. */
    priors?: number[];
    /** Dimensionality for `transform`; at most min(nClasses - 1, nFeatures). */
    nComponents?: number;
    /** Rank-estimation threshold for the SVD solver. */
    tol?: number;
}

/**
 * Linear Discriminant Analysis matching sklearn's
 * `LinearDiscriminantAnalysis` for the 'svd' and 'eigen' solvers: a linear
 * classifier with class-conditional Gaussian densities sharing one covariance
 * matrix, that doubles as a supervised dimensionality reduction (transform
 * projects onto the most discriminative axes).
 */
export class LinearDiscriminantAnalysis extends ClassifierBase {
    private solver: LDASolver;
    private shrinkage?: number;
    private priors?: number[];
    private nComponents?: number;
    private tol: number;

    private classes: number[];
    private classPriors: number[];
    private means: number[][];
    private xbar: number[];
    /** projection matrix, nFeatures x nDiscriminantAxes */
    private scalings: number[][];
    /** per-class linear scores: coef[k] . x + intercept[k], shape [nClasses][nFeatures] */
    private coef: number[][];
    private intercept: number[];
    private explainedVarianceRatio: number[];
    private maxComponents: number;
    private nFeatures: number;
    private fitted: boolean;

    constructor(props: LinearDiscriminantAnalysisProps = {}) {
        super();
        const { solver = 'svd', shrinkage, priors, nComponents, tol = 1e-4 } = props;
        this.solver = solver;
        this.shrinkage = shrinkage;
        this.priors = priors == null ? undefined : priors.slice();
        this.nComponents = nComponents;
        this.tol = tol;

        this.classes = [];
        this.classPriors = [];
        this.means = [];
        this.xbar = [];
        this.scalings = [];
        this.coef = [];
        this.intercept = [];
        this.explainedVarianceRatio = [];
        this.maxComponents = 0;
        this.nFeatures = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            solver: this.solver,
            shrinkage: this.shrinkage,
            priors: this.priors == null ? undefined : this.priors.slice(),
            nComponents: this.nComponents,
            tol: this.tol,
        };
    }

    public fit(X: number[][], y: number[]): void {
        validateXy(X, y, 'LinearDiscriminantAnalysis');
        if (this.solver !== 'svd' && this.solver !== 'eigen') {
            throw new Error(`LinearDiscriminantAnalysis: unknown solver "${this.solver}" (use 'svd' or 'eigen')`);
        }
        if (this.shrinkage != null) {
            if (this.solver === 'svd') {
                throw new Error("LinearDiscriminantAnalysis: shrinkage not supported with 'svd' solver; use solver='eigen'");
            }
            if (!Number.isFinite(this.shrinkage) || this.shrinkage < 0 || this.shrinkage > 1) {
                throw new Error('LinearDiscriminantAnalysis: shrinkage must be between 0 and 1');
            }
        }

        const n = X.length;
        const p = X[0].length;
        const classes = sortedUniqueLabels(y);
        const K = classes.length;
        if (K < 2) {
            throw new Error(`LinearDiscriminantAnalysis: y must contain at least 2 classes, got ${K}`);
        }
        const labelToIdx = new Map<number, number>();
        classes.forEach((c, i) => labelToIdx.set(c, i));
        const classIdx = y.map((v) => labelToIdx.get(v)!);
        const counts = new Array<number>(K).fill(0);
        for (const k of classIdx) counts[k] += 1;

        this.classes = classes;
        this.nFeatures = p;
        this.classPriors = resolvePriors(this.priors, counts, n, 'LinearDiscriminantAnalysis');
        this.means = classMeans(X, classIdx, K);
        this.maxComponents = Math.min(K - 1, p);
        if (this.nComponents != null) {
            if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
                throw new Error('LinearDiscriminantAnalysis: nComponents must be a positive integer');
            }
            if (this.nComponents > this.maxComponents) {
                throw new Error(
                    `LinearDiscriminantAnalysis: nComponents cannot be larger than min(nFeatures, nClasses - 1) = ${this.maxComponents}`
                );
            }
        }

        if (this.solver === 'svd') {
            this.solveSVD(X, classIdx);
        } else {
            this.solveEigen(X, classIdx);
        }
        this.fitted = true;
    }

    /**
     * sklearn's `_solve_svd`: never forms a covariance matrix. Std-scale the
     * class-centered data, SVD it to whiten within-class directions (rank cut
     * at `tol`), then SVD the whitened between-class means.
     */
    private solveSVD(X: number[][], classIdx: number[]): void {
        const n = X.length;
        const p = X[0].length;
        const K = this.classes.length;
        if (n <= K) {
            throw new Error(
                "LinearDiscriminantAnalysis: the 'svd' solver requires nSamples > nClasses"
            );
        }

        // weighted overall mean
        const xbar = new Array<number>(p).fill(0);
        for (let k = 0; k < K; k++) {
            for (let j = 0; j < p; j++) xbar[j] += this.classPriors[k] * this.means[k][j];
        }
        this.xbar = xbar;

        // class-centered data
        const Xc: number[][] = X.map((row, i) => {
            const m = this.means[classIdx[i]];
            return row.map((v, j) => v - m[j]);
        });

        // per-feature std of Xc (population, ddof=0); zeros replaced by 1
        const std = new Array<number>(p).fill(0);
        const colMean = new Array<number>(p).fill(0);
        for (const row of Xc) for (let j = 0; j < p; j++) colMean[j] += row[j];
        for (let j = 0; j < p; j++) colMean[j] /= n;
        for (const row of Xc) {
            for (let j = 0; j < p; j++) {
                const d = row[j] - colMean[j];
                std[j] += d * d;
            }
        }
        for (let j = 0; j < p; j++) {
            std[j] = Math.sqrt(std[j] / n);
            if (std[j] === 0) std[j] = 1;
        }

        const fac = 1 / (n - K);
        const sqrtFac = Math.sqrt(fac);
        const Xsc = Xc.map((row) => row.map((v, j) => (sqrtFac * v) / std[j]));

        // 1) whiten within-class variation
        const { S, Vt } = jacobiSVD(Xsc);
        let rank = 0;
        while (rank < S.length && S[rank] > this.tol) rank++;
        // scalings0: p x rank, column r = Vt[r] / std / S[r]
        const scalings0: number[][] = [];
        for (let j = 0; j < p; j++) {
            const row = new Array<number>(rank);
            for (let r = 0; r < rank; r++) row[r] = Vt[r][j] / std[j] / S[r];
            scalings0.push(row);
        }

        // 2) SVD of the whitened, prior-weighted between-class means
        const X2: number[][] = [];
        for (let k = 0; k < K; k++) {
            const w = Math.sqrt(n * this.classPriors[k] * fac);
            const row = new Array<number>(rank).fill(0);
            for (let r = 0; r < rank; r++) {
                let s = 0;
                for (let j = 0; j < p; j++) s += (this.means[k][j] - xbar[j]) * scalings0[j][r];
                row[r] = w * s;
            }
            X2.push(row);
        }
        const { S: S2, Vt: Vt2 } = jacobiSVD(X2);

        let totalS2 = 0;
        for (const s of S2) totalS2 += s * s;
        this.explainedVarianceRatio = S2
            .map((s) => (totalS2 > 0 ? (s * s) / totalS2 : 0))
            .slice(0, this.maxComponents);

        let rank2 = 0;
        const s2Head = S2.length > 0 ? S2[0] : 0;
        while (rank2 < S2.length && S2[rank2] > this.tol * s2Head) rank2++;

        // scalings = scalings0 @ Vt2^T[:, :rank2]  (p x rank2)
        const scalings: number[][] = [];
        for (let j = 0; j < p; j++) {
            const row = new Array<number>(rank2).fill(0);
            for (let c = 0; c < rank2; c++) {
                let s = 0;
                for (let r = 0; r < rank; r++) s += scalings0[j][r] * Vt2[c][r];
                row[c] = s;
            }
            scalings.push(row);
        }
        this.scalings = scalings;

        // linear scores in the discriminant space, folded back to input space
        const coefD: number[][] = [];
        for (let k = 0; k < K; k++) {
            const row = new Array<number>(rank2).fill(0);
            for (let c = 0; c < rank2; c++) {
                let s = 0;
                for (let j = 0; j < p; j++) s += (this.means[k][j] - xbar[j]) * scalings[j][c];
                row[c] = s;
            }
            coefD.push(row);
        }
        const intercept = new Array<number>(K).fill(0);
        const coef: number[][] = [];
        for (let k = 0; k < K; k++) {
            let sq = 0;
            for (const v of coefD[k]) sq += v * v;
            intercept[k] = -0.5 * sq + Math.log(this.classPriors[k]);
            const row = new Array<number>(p).fill(0);
            for (let j = 0; j < p; j++) {
                let s = 0;
                for (let c = 0; c < coefD[k].length; c++) s += coefD[k][c] * scalings[j][c];
                row[j] = s;
            }
            coef.push(row);
            let xb = 0;
            for (let j = 0; j < p; j++) xb += xbar[j] * row[j];
            intercept[k] -= xb;
        }
        this.coef = coef;
        this.intercept = intercept;
    }

    /** Biased covariance (divide by n) with optional shrinkage toward mean-eigenvalue identity. */
    private static shrunkCovariance(rows: number[][], center: number[], shrinkage: number | undefined): number[][] {
        const n = rows.length;
        const p = center.length;
        const cov: number[][] = [];
        for (let a = 0; a < p; a++) cov.push(new Array<number>(p).fill(0));
        for (const row of rows) {
            for (let a = 0; a < p; a++) {
                const da = row[a] - center[a];
                for (let b = a; b < p; b++) {
                    cov[a][b] += da * (row[b] - center[b]);
                }
            }
        }
        for (let a = 0; a < p; a++) {
            for (let b = a; b < p; b++) {
                cov[a][b] /= n;
                cov[b][a] = cov[a][b];
            }
        }
        if (shrinkage != null && shrinkage > 0) {
            let mu = 0;
            for (let a = 0; a < p; a++) mu += cov[a][a];
            mu /= p;
            for (let a = 0; a < p; a++) {
                for (let b = 0; b < p; b++) {
                    cov[a][b] *= 1 - shrinkage;
                }
                cov[a][a] += shrinkage * mu;
            }
        }
        return cov;
    }

    /**
     * sklearn's `_solve_eigen`: generalized eigenproblem Sb v = lambda Sw v,
     * solved via Cholesky of Sw (eigenvectors normalized so v' Sw v = 1,
     * matching scipy.linalg.eigh).
     */
    private solveEigen(X: number[][], classIdx: number[]): void {
        const n = X.length;
        const p = X[0].length;
        const K = this.classes.length;

        // within-class covariance: prior-weighted average of per-class (shrunk) covariances
        const Sw: number[][] = [];
        for (let a = 0; a < p; a++) Sw.push(new Array<number>(p).fill(0));
        for (let k = 0; k < K; k++) {
            const rows: number[][] = [];
            for (let i = 0; i < n; i++) if (classIdx[i] === k) rows.push(X[i]);
            const covK = LinearDiscriminantAnalysis.shrunkCovariance(rows, this.means[k], this.shrinkage);
            for (let a = 0; a < p; a++) {
                for (let b = 0; b < p; b++) Sw[a][b] += this.classPriors[k] * covK[a][b];
            }
        }

        // total covariance (same shrinkage), between = total - within
        const grandMean = new Array<number>(p).fill(0);
        for (const row of X) for (let j = 0; j < p; j++) grandMean[j] += row[j];
        for (let j = 0; j < p; j++) grandMean[j] /= n;
        const St = LinearDiscriminantAnalysis.shrunkCovariance(X, grandMean, this.shrinkage);
        const Sb: number[][] = [];
        for (let a = 0; a < p; a++) {
            Sb.push(new Array<number>(p).fill(0));
            for (let b = 0; b < p; b++) Sb[a][b] = St[a][b] - Sw[a][b];
        }

        const L = cholesky(Sw);
        if (L === null) {
            throw new Error(
                "LinearDiscriminantAnalysis: within-class covariance is singular; use shrinkage or the 'svd' solver"
            );
        }
        // C = L^-1 Sb L^-T (symmetric); row i of C = L^-1 (row i of (L^-1 Sb)^T)
        const Y: number[][] = []; // Y = L^-1 Sb, built column by column
        for (let a = 0; a < p; a++) Y.push(new Array<number>(p).fill(0));
        for (let j = 0; j < p; j++) {
            const col = Sb.map((row) => row[j]);
            const sol = solveLower(L, col);
            for (let a = 0; a < p; a++) Y[a][j] = sol[a];
        }
        const C: number[][] = [];
        for (let i = 0; i < p; i++) C.push(solveLower(L, Y[i]));
        for (let a = 0; a < p; a++) {
            for (let b = a + 1; b < p; b++) {
                const m = (C[a][b] + C[b][a]) / 2;
                C[a][b] = m;
                C[b][a] = m;
            }
        }

        const { values, vectors } = symmetricEigDecomposition(C);
        // generalized eigenvectors, normalized so v' Sw v = 1
        const evecs = vectors.map((u) => solveLowerTransposed(L, u));

        let total = 0;
        for (const v of values) total += v;
        this.explainedVarianceRatio = values
            .map((v) => (total !== 0 ? v / total : 0))
            .slice(0, this.maxComponents);

        // scalings: p x p, column i = eigenvector i (descending eigenvalue)
        const scalings: number[][] = [];
        for (let j = 0; j < p; j++) {
            const row = new Array<number>(evecs.length);
            for (let i = 0; i < evecs.length; i++) row[i] = evecs[i][j];
            scalings.push(row);
        }
        this.scalings = scalings;
        this.xbar = new Array<number>(p).fill(0); // eigen transform does not center (sklearn parity)

        // coef = means @ evecs @ evecs^T ; intercept = -0.5 diag(means coef^T) + log priors
        const coef: number[][] = [];
        const intercept = new Array<number>(K).fill(0);
        for (let k = 0; k < K; k++) {
            const proj = evecs.map((v) => {
                let s = 0;
                for (let j = 0; j < p; j++) s += this.means[k][j] * v[j];
                return s;
            });
            const row = new Array<number>(p).fill(0);
            for (let j = 0; j < p; j++) {
                let s = 0;
                for (let i = 0; i < evecs.length; i++) s += proj[i] * evecs[i][j];
                row[j] = s;
            }
            coef.push(row);
            let mk = 0;
            for (let j = 0; j < p; j++) mk += this.means[k][j] * row[j];
            intercept[k] = -0.5 * mk + Math.log(this.classPriors[k]);
        }
        this.coef = coef;
        this.intercept = intercept;
    }

    private assertFitted(method: string): void {
        if (!this.fitted) {
            throw new Error(`LinearDiscriminantAnalysis must be fitted before calling ${method}`);
        }
    }

    /** Per-class linear scores, shape [nSamples][nClasses]. */
    private decisionScores(X: number[][]): number[][] {
        validatePredictInput(X, this.nFeatures, 'LinearDiscriminantAnalysis');
        return X.map((row) => {
            return this.coef.map((c, k) => {
                let s = this.intercept[k];
                for (let j = 0; j < row.length; j++) s += c[j] * row[j];
                return s;
            });
        });
    }

    public predict(testX: number[][]): number[] {
        this.assertFitted('predict');
        return this.decisionScores(testX).map((row) => this.classes[argmaxRow(row)]);
    }

    /** Softmax over the per-class linear scores (sklearn parity). */
    public predictProba(testX: number[][]): number[][] {
        this.assertFitted('predictProba');
        return softmaxRows(this.decisionScores(testX));
    }

    /**
     * Binary: 1-D signed scores for `classes[1]` (sklearn convention).
     * Multiclass: per-class scores, shape [nSamples][nClasses].
     */
    public decisionFunction(testX: number[][]): number[] | number[][] {
        this.assertFitted('decisionFunction');
        const scores = this.decisionScores(testX);
        if (this.classes.length === 2) {
            return scores.map((row) => row[1] - row[0]);
        }
        return scores;
    }

    /** Project data onto the discriminant axes (at most nComponents columns). */
    public transform(X: number[][]): number[][] {
        this.assertFitted('transform');
        validatePredictInput(X, this.nFeatures, 'LinearDiscriminantAnalysis');
        const available = this.scalings.length > 0 ? this.scalings[0].length : 0;
        const width = Math.min(this.nComponents ?? this.maxComponents, available);
        return X.map((row) => {
            const out = new Array<number>(width).fill(0);
            for (let c = 0; c < width; c++) {
                let s = 0;
                for (let j = 0; j < row.length; j++) {
                    s += (row[j] - this.xbar[j]) * this.scalings[j][c];
                }
                out[c] = s;
            }
            return out;
        });
    }

    public fitTransform(X: number[][], y: number[]): number[][] {
        this.fit(X, y);
        return this.transform(X);
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    /** Weight matrix in sklearn shape: [1][nFeatures] for binary, [nClasses][nFeatures] otherwise. */
    public getCoef(): number[][] {
        this.assertFitted('getCoef');
        if (this.classes.length === 2) {
            return [this.coef[1].map((v, j) => v - this.coef[0][j])];
        }
        return this.coef.map((row) => row.slice());
    }

    /** Intercepts in sklearn shape: length 1 for binary, nClasses otherwise. */
    public getIntercept(): number[] {
        this.assertFitted('getIntercept');
        if (this.classes.length === 2) {
            return [this.intercept[1] - this.intercept[0]];
        }
        return this.intercept.slice();
    }

    public getMeans(): number[][] {
        this.assertFitted('getMeans');
        return this.means.map((row) => row.slice());
    }

    public getPriors(): number[] {
        this.assertFitted('getPriors');
        return this.classPriors.slice();
    }

    public getScalings(): number[][] {
        this.assertFitted('getScalings');
        return this.scalings.map((row) => row.slice());
    }

    public getExplainedVarianceRatio(): number[] {
        this.assertFitted('getExplainedVarianceRatio');
        return this.explainedVarianceRatio.slice();
    }
}
registerEstimator('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis);
