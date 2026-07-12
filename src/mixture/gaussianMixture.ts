/**
 * Gaussian Mixture Model fitted with full expectation-maximization,
 * mirroring sklearn.mixture.GaussianMixture (camelCase params).
 *
 * The E-step works entirely in the log domain (logsumexp) so responsibilities
 * never underflow; the M-step re-estimates weights/means/covariances per
 * covariance type with `regCovar` added to diagonals. Convergence is declared
 * when the change in average per-sample log-likelihood falls below `tol`;
 * the best of `nInit` runs (highest lower bound) is kept.
 */
import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import { logsumexp, spdInverse } from './math';
import {
    CovarianceType,
    Covariances,
    estimateGaussianParameters,
    estimateLogGaussianProb,
    initializeResponsibilities,
} from './common';

export interface GaussianMixtureProps {
    /** number of mixture components (default 1) */
    nComponents?: number;
    /** covariance parametrization: 'full' | 'tied' | 'diag' | 'spherical' (default 'full') */
    covarianceType?: CovarianceType;
    /** EM convergence threshold on the average log-likelihood change (default 1e-3) */
    tol?: number;
    /** non-negative regularization added to covariance diagonals (default 1e-6) */
    regCovar?: number;
    /** maximum number of EM iterations (default 100) */
    maxIter?: number;
    /** number of EM restarts; the run with the best log-likelihood wins (default 1) */
    nInit?: number;
    /** responsibility initialization: 'kmeans' | 'random' (default 'kmeans') */
    initParams?: 'kmeans' | 'random';
    /** seed for reproducible initialization */
    randomState?: number;
    /** user-provided initial weights (length nComponents) */
    weightsInit?: number[];
    /** user-provided initial means (nComponents × nFeatures) */
    meansInit?: number[][];
    /**
     * user-provided initial precisions, shaped per covarianceType:
     * 'full' k×d×d, 'tied' d×d, 'diag' k×d, 'spherical' k
     */
    precisionsInit?: Covariances;
}

const COVARIANCE_TYPES: CovarianceType[] = ['full', 'tied', 'diag', 'spherical'];

interface EMRunResult {
    weights: number[];
    means: number[][];
    covariances: Covariances;
    lowerBound: number;
    lowerBoundHistory: number[];
    converged: boolean;
    nIter: number;
}

export class GaussianMixture extends ClusterBase {
    private nComponents: number;
    private covarianceType: CovarianceType;
    private tol: number;
    private regCovar: number;
    private maxIter: number;
    private nInit: number;
    private initParams: 'kmeans' | 'random';
    private randomState?: number;
    private weightsInit: number[] | null;
    private meansInit: number[][] | null;
    private precisionsInit: Covariances | null;

    private weights_: number[] | null;
    private means_: number[][] | null;
    private covariances_: Covariances | null;
    /** whether the best EM run reached the `tol` criterion */
    public converged: boolean;
    /** number of EM iterations performed by the best run */
    public nIter: number;
    private lowerBound_: number;
    private lowerBoundHistory_: number[];

    constructor(props: GaussianMixtureProps = {}) {
        super();
        const {
            nComponents = 1,
            covarianceType = 'full',
            tol = 1e-3,
            regCovar = 1e-6,
            maxIter = 100,
            nInit = 1,
            initParams = 'kmeans',
            randomState,
            weightsInit,
            meansInit,
            precisionsInit,
        } = props;
        if (!Number.isInteger(nComponents) || nComponents < 1) {
            throw new Error(`nComponents must be a positive integer (got ${nComponents})`);
        }
        if (!COVARIANCE_TYPES.includes(covarianceType)) {
            throw new Error(`Invalid covarianceType "${covarianceType}". Valid types: ${COVARIANCE_TYPES.join(', ')}.`);
        }
        if (!(tol >= 0)) throw new Error(`tol must be non-negative (got ${tol})`);
        if (!(regCovar >= 0)) throw new Error(`regCovar must be non-negative (got ${regCovar})`);
        if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error(`maxIter must be a positive integer (got ${maxIter})`);
        if (!Number.isInteger(nInit) || nInit < 1) throw new Error(`nInit must be a positive integer (got ${nInit})`);
        if (initParams !== 'kmeans' && initParams !== 'random') {
            throw new Error(`Invalid initParams "${initParams}". Valid values: kmeans, random.`);
        }
        if (weightsInit && weightsInit.length !== nComponents) {
            throw new Error(`weightsInit must have length nComponents=${nComponents} (got ${weightsInit.length})`);
        }
        if (meansInit && meansInit.length !== nComponents) {
            throw new Error(`meansInit must have nComponents=${nComponents} rows (got ${meansInit.length})`);
        }
        this.nComponents = nComponents;
        this.covarianceType = covarianceType;
        this.tol = tol;
        this.regCovar = regCovar;
        this.maxIter = maxIter;
        this.nInit = nInit;
        this.initParams = initParams;
        this.randomState = randomState;
        this.weightsInit = weightsInit ? weightsInit.slice() : null;
        this.meansInit = meansInit ? meansInit.map((row) => row.slice()) : null;
        this.precisionsInit = precisionsInit ? (JSON.parse(JSON.stringify(precisionsInit)) as Covariances) : null;
        this.weights_ = null;
        this.means_ = null;
        this.covariances_ = null;
        this.converged = false;
        this.nIter = 0;
        this.lowerBound_ = -Infinity;
        this.lowerBoundHistory_ = [];
    }

    public getParams(): Params {
        return {
            nComponents: this.nComponents,
            covarianceType: this.covarianceType,
            tol: this.tol,
            regCovar: this.regCovar,
            maxIter: this.maxIter,
            nInit: this.nInit,
            initParams: this.initParams,
            randomState: this.randomState,
            weightsInit: this.weightsInit ?? undefined,
            meansInit: this.meansInit ?? undefined,
            precisionsInit: this.precisionsInit ?? undefined,
        };
    }

    /** convert user-provided precisions into the covariance container the E-step consumes */
    private precisionsToCovariances(precisions: Covariances, nFeatures: number): Covariances {
        switch (this.covarianceType) {
            case 'full':
                return (precisions as number[][][]).map((p) => spdInverse(p));
            case 'tied':
                return spdInverse(precisions as number[][]);
            case 'diag':
                return (precisions as number[][]).map((row) => row.map((v) => 1 / v));
            case 'spherical':
                return (precisions as number[]).map((v) => 1 / v);
        }
    }

    private initializeParameters(
        X: number[][],
        rng: () => number
    ): { weights: number[]; means: number[][]; covariances: Covariances } {
        const n = X.length;
        const resp = initializeResponsibilities(X, this.nComponents, this.initParams, rng);
        const { nk, means, covariances } = estimateGaussianParameters(X, resp, this.regCovar, this.covarianceType);
        let weights = nk.map((v) => v / n);
        let outMeans = means;
        let outCovariances = covariances;
        if (this.weightsInit) weights = this.weightsInit.slice();
        if (this.meansInit) outMeans = this.meansInit.map((row) => row.slice());
        if (this.precisionsInit) outCovariances = this.precisionsToCovariances(this.precisionsInit, X[0].length);
        return { weights, means: outMeans, covariances: outCovariances };
    }

    /** n × k matrix of log(weightₖ) + log N(xᵢ | μₖ, Σₖ) */
    private static weightedLogProb(
        X: number[][],
        weights: number[],
        means: number[][],
        covariances: Covariances,
        covarianceType: CovarianceType
    ): number[][] {
        const logProb = estimateLogGaussianProb(X, means, covariances, covarianceType);
        const logWeights = weights.map((w) => Math.log(w));
        for (let i = 0; i < logProb.length; i++) {
            for (let c = 0; c < logWeights.length; c++) logProb[i][c] += logWeights[c];
        }
        return logProb;
    }

    private runEM(X: number[][], rng: () => number): EMRunResult {
        const n = X.length;
        let { weights, means, covariances } = this.initializeParameters(X, rng);
        let lowerBound = -Infinity;
        let converged = false;
        let nIter = 0;
        const history: number[] = [];
        for (let iter = 1; iter <= this.maxIter; iter++) {
            const prevLowerBound = lowerBound;
            // ---- E-step (log domain) ----
            const weighted = GaussianMixture.weightedLogProb(X, weights, means, covariances, this.covarianceType);
            let logProbNormSum = 0;
            const resp: number[][] = [];
            for (let i = 0; i < n; i++) {
                const norm = logsumexp(weighted[i]);
                logProbNormSum += norm;
                const row = new Array(this.nComponents).fill(0);
                for (let c = 0; c < this.nComponents; c++) row[c] = Math.exp(weighted[i][c] - norm);
                resp.push(row);
            }
            lowerBound = logProbNormSum / n;
            history.push(lowerBound);
            nIter = iter;
            // ---- M-step ----
            const stats = estimateGaussianParameters(X, resp, this.regCovar, this.covarianceType);
            weights = stats.nk.map((v) => v / n);
            means = stats.means;
            covariances = stats.covariances;
            if (prevLowerBound !== -Infinity && Math.abs(lowerBound - prevLowerBound) < this.tol) {
                converged = true;
                break;
            }
        }
        return { weights, means, covariances, lowerBound, lowerBoundHistory: history, converged, nIter };
    }

    public fit(X: number[][]): this {
        if (!X || X.length === 0) throw new Error('GaussianMixture.fit requires a non-empty X');
        if (X.length < this.nComponents) {
            throw new Error(`Expected n_samples >= nComponents but got nComponents = ${this.nComponents}, n_samples = ${X.length}`);
        }
        const rng = createRandomGenerator(this.randomState);
        let best: EMRunResult | null = null;
        for (let run = 0; run < this.nInit; run++) {
            const result = this.runEM(X, rng);
            if (best === null || result.lowerBound > best.lowerBound) best = result;
        }
        this.weights_ = best!.weights;
        this.means_ = best!.means;
        this.covariances_ = best!.covariances;
        this.converged = best!.converged;
        this.nIter = best!.nIter;
        this.lowerBound_ = best!.lowerBound;
        this.lowerBoundHistory_ = best!.lowerBoundHistory;
        return this;
    }

    private checkFitted(): void {
        if (this.weights_ === null || this.means_ === null || this.covariances_ === null) {
            throw new Error('This GaussianMixture instance is not fitted yet. Call fit(X) first.');
        }
    }

    /** hard assignment: argmax of the responsibilities */
    public predict(X: number[][]): number[] {
        this.checkFitted();
        const weighted = GaussianMixture.weightedLogProb(X, this.weights_!, this.means_!, this.covariances_!, this.covarianceType);
        return weighted.map((row) => {
            let arg = 0;
            for (let c = 1; c < row.length; c++) {
                if (row[c] > row[arg]) arg = c;
            }
            return arg;
        });
    }

    /** posterior probability (responsibility) of each component for each sample */
    public predictProba(X: number[][]): number[][] {
        this.checkFitted();
        const weighted = GaussianMixture.weightedLogProb(X, this.weights_!, this.means_!, this.covariances_!, this.covarianceType);
        return weighted.map((row) => {
            const norm = logsumexp(row);
            return row.map((v) => Math.exp(v - norm));
        });
    }

    public fitPredict(X: number[][]): number[] {
        this.fit(X);
        return this.predict(X);
    }

    /** per-sample log-likelihood log p(xᵢ) under the mixture */
    public scoreSamples(X: number[][]): number[] {
        this.checkFitted();
        const weighted = GaussianMixture.weightedLogProb(X, this.weights_!, this.means_!, this.covariances_!, this.covarianceType);
        return weighted.map((row) => logsumexp(row));
    }

    /** average per-sample log-likelihood */
    public score(X: number[][]): number {
        const scores = this.scoreSamples(X);
        return scores.reduce((s, v) => s + v, 0) / scores.length;
    }

    /** number of free parameters, counted exactly like sklearn's `_n_parameters` */
    private nParameters(): number {
        this.checkFitted();
        const k = this.nComponents;
        const d = this.means_![0].length;
        let covParams: number;
        switch (this.covarianceType) {
            case 'full':
                covParams = (k * d * (d + 1)) / 2;
                break;
            case 'tied':
                covParams = (d * (d + 1)) / 2;
                break;
            case 'diag':
                covParams = k * d;
                break;
            case 'spherical':
                covParams = k;
                break;
        }
        return covParams + k * d + (k - 1);
    }

    /** Akaike information criterion: −2·logL + 2·nParams (lower is better) */
    public aic(X: number[][]): number {
        return -2 * this.score(X) * X.length + 2 * this.nParameters();
    }

    /** Bayesian information criterion: −2·logL + ln(n)·nParams (lower is better) */
    public bic(X: number[][]): number {
        return -2 * this.score(X) * X.length + Math.log(X.length) * this.nParameters();
    }

    public getWeights(): number[] | null {
        return this.weights_;
    }

    public getMeans(): number[][] | null {
        return this.means_;
    }

    public getCovariances(): Covariances | null {
        return this.covariances_;
    }

    /** average per-sample log-likelihood of the best EM run at its final E-step */
    public getLowerBound(): number {
        return this.lowerBound_;
    }

    /** lower-bound trajectory of the best EM run (one entry per iteration, non-decreasing) */
    public getLowerBoundHistory(): number[] {
        return this.lowerBoundHistory_.slice();
    }
}
registerEstimator('GaussianMixture', GaussianMixture);
