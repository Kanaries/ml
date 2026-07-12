/**
 * Variational Bayesian estimation of a Gaussian mixture, following the update
 * equations of sklearn.mixture.BayesianGaussianMixture (Bishop, PRML ch. 10.2):
 *
 *  - weights carry a Dirichlet prior ('dirichletDistribution') or a
 *    stick-breaking Dirichlet-process prior ('dirichletProcess'), updated with
 *    digamma expectations;
 *  - each component's mean/precision carries a Gaussian-Wishart prior updated
 *    from weighted sufficient statistics;
 *  - the E-step uses the expected log-likelihood (digamma-corrected Gaussian
 *    densities), and convergence is declared when the change in the variational
 *    lower bound falls below `tol`.
 *
 * covarianceType 'full' and 'diag' are implemented; 'tied' and 'spherical'
 * throw a clear not-implemented error.
 */
import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import { betaln, digamma, logGamma, logsumexp } from './math';
import {
    computeLogDetCholesky,
    estimateGaussianParameters,
    estimateLogGaussianProb,
    initializeResponsibilities,
} from './common';

export type BayesianCovarianceType = 'full' | 'diag';
export type WeightConcentrationPriorType = 'dirichletProcess' | 'dirichletDistribution';

export interface BayesianGaussianMixtureProps {
    /** number of mixture components; with a DP prior extra ones shrink to ~0 weight (default 1) */
    nComponents?: number;
    /** covariance parametrization: 'full' | 'diag' (default 'full') */
    covarianceType?: BayesianCovarianceType;
    /** convergence threshold on the lower-bound change (default 1e-3) */
    tol?: number;
    /** non-negative regularization added to covariance diagonals (default 1e-6) */
    regCovar?: number;
    /** maximum number of variational iterations (default 100) */
    maxIter?: number;
    /** number of restarts; the run with the best lower bound wins (default 1) */
    nInit?: number;
    /** responsibility initialization: 'kmeans' | 'random' (default 'kmeans') */
    initParams?: 'kmeans' | 'random';
    /** seed for reproducible initialization */
    randomState?: number;
    /** prior family on the weights (default 'dirichletProcess') */
    weightConcentrationPriorType?: WeightConcentrationPriorType;
    /** Dirichlet/stick-breaking concentration; default 1 / nComponents */
    weightConcentrationPrior?: number;
    /** precision prior on the mean distributions (default 1) */
    meanPrecisionPrior?: number;
    /** prior on the mean distributions (default: column means of X) */
    meanPrior?: number[];
    /** Wishart degrees-of-freedom prior; must exceed nFeatures − 1 (default nFeatures) */
    degreesOfFreedomPrior?: number;
    /** covariance prior: d×d matrix for 'full', length-d variance vector for 'diag' (default: empirical) */
    covariancePrior?: number[][] | number[];
}

type BayesianCovariances = number[][][] | number[][];

interface VBPriors {
    wcPrior: number;
    mpPrior: number;
    meanPrior: number[];
    dofPrior: number;
    covPrior: number[][] | number[];
}

interface VBState {
    /** first Beta parameter (DP) or Dirichlet concentration (dirichletDistribution) */
    wcA: number[];
    /** second Beta parameter (DP) or null (dirichletDistribution) */
    wcB: number[] | null;
    meanPrecision: number[];
    means: number[][];
    dof: number[];
    covariances: BayesianCovariances;
}

interface VBRunResult extends VBState {
    lowerBound: number;
    lowerBoundHistory: number[];
    converged: boolean;
    nIter: number;
}

const SUPPORTED_COVARIANCE_TYPES: BayesianCovarianceType[] = ['full', 'diag'];

export class BayesianGaussianMixture extends ClusterBase {
    private nComponents: number;
    private covarianceType: BayesianCovarianceType;
    private tol: number;
    private regCovar: number;
    private maxIter: number;
    private nInit: number;
    private initParams: 'kmeans' | 'random';
    private randomState?: number;
    private weightConcentrationPriorType: WeightConcentrationPriorType;
    private weightConcentrationPrior?: number;
    private meanPrecisionPrior?: number;
    private meanPrior: number[] | null;
    private degreesOfFreedomPrior?: number;
    private covariancePrior: number[][] | number[] | null;

    private weights_: number[] | null;
    private weightConcentrationA_: number[] | null;
    private weightConcentrationB_: number[] | null;
    private meanPrecision_: number[] | null;
    private means_: number[][] | null;
    private degreesOfFreedom_: number[] | null;
    private covariances_: BayesianCovariances | null;
    /** whether the best run reached the `tol` criterion */
    public converged: boolean;
    /** number of variational iterations performed by the best run */
    public nIter: number;
    private lowerBound_: number;
    private lowerBoundHistory_: number[];

    constructor(props: BayesianGaussianMixtureProps = {}) {
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
            weightConcentrationPriorType = 'dirichletProcess',
            weightConcentrationPrior,
            meanPrecisionPrior,
            meanPrior,
            degreesOfFreedomPrior,
            covariancePrior,
        } = props;
        if (!Number.isInteger(nComponents) || nComponents < 1) {
            throw new Error(`nComponents must be a positive integer (got ${nComponents})`);
        }
        if (!SUPPORTED_COVARIANCE_TYPES.includes(covarianceType)) {
            throw new Error(
                `covarianceType "${covarianceType}" is not implemented for BayesianGaussianMixture. ` +
                    `Supported types: ${SUPPORTED_COVARIANCE_TYPES.join(', ')}.`
            );
        }
        if (!(tol >= 0)) throw new Error(`tol must be non-negative (got ${tol})`);
        if (!(regCovar >= 0)) throw new Error(`regCovar must be non-negative (got ${regCovar})`);
        if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error(`maxIter must be a positive integer (got ${maxIter})`);
        if (!Number.isInteger(nInit) || nInit < 1) throw new Error(`nInit must be a positive integer (got ${nInit})`);
        if (initParams !== 'kmeans' && initParams !== 'random') {
            throw new Error(`Invalid initParams "${initParams}". Valid values: kmeans, random.`);
        }
        if (weightConcentrationPriorType !== 'dirichletProcess' && weightConcentrationPriorType !== 'dirichletDistribution') {
            throw new Error(
                `Invalid weightConcentrationPriorType "${weightConcentrationPriorType}". ` +
                    'Valid values: dirichletProcess, dirichletDistribution.'
            );
        }
        if (weightConcentrationPrior !== undefined && !(weightConcentrationPrior > 0)) {
            throw new Error(`weightConcentrationPrior must be positive (got ${weightConcentrationPrior})`);
        }
        if (meanPrecisionPrior !== undefined && !(meanPrecisionPrior > 0)) {
            throw new Error(`meanPrecisionPrior must be positive (got ${meanPrecisionPrior})`);
        }
        this.nComponents = nComponents;
        this.covarianceType = covarianceType;
        this.tol = tol;
        this.regCovar = regCovar;
        this.maxIter = maxIter;
        this.nInit = nInit;
        this.initParams = initParams;
        this.randomState = randomState;
        this.weightConcentrationPriorType = weightConcentrationPriorType;
        this.weightConcentrationPrior = weightConcentrationPrior;
        this.meanPrecisionPrior = meanPrecisionPrior;
        this.meanPrior = meanPrior ? meanPrior.slice() : null;
        this.degreesOfFreedomPrior = degreesOfFreedomPrior;
        this.covariancePrior = covariancePrior ? (JSON.parse(JSON.stringify(covariancePrior)) as number[][] | number[]) : null;
        this.weights_ = null;
        this.weightConcentrationA_ = null;
        this.weightConcentrationB_ = null;
        this.meanPrecision_ = null;
        this.means_ = null;
        this.degreesOfFreedom_ = null;
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
            weightConcentrationPriorType: this.weightConcentrationPriorType,
            weightConcentrationPrior: this.weightConcentrationPrior,
            meanPrecisionPrior: this.meanPrecisionPrior,
            meanPrior: this.meanPrior ?? undefined,
            degreesOfFreedomPrior: this.degreesOfFreedomPrior,
            covariancePrior: this.covariancePrior ?? undefined,
        };
    }

    // -----------------------------------------------------------------------
    // priors
    // -----------------------------------------------------------------------

    private resolvePriors(X: number[][]): VBPriors {
        const n = X.length;
        const d = X[0].length;
        const colMeans = new Array(d).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) colMeans[j] += X[i][j];
        }
        for (let j = 0; j < d; j++) colMeans[j] /= n;

        const dofPrior = this.degreesOfFreedomPrior ?? d;
        if (!(dofPrior > d - 1)) {
            throw new Error(`degreesOfFreedomPrior must be greater than nFeatures - 1 = ${d - 1} (got ${dofPrior})`);
        }
        let covPrior: number[][] | number[];
        if (this.covariancePrior) {
            covPrior = JSON.parse(JSON.stringify(this.covariancePrior)) as number[][] | number[];
        } else {
            const ddof = Math.max(1, n - 1);
            if (this.covarianceType === 'full') {
                // empirical covariance of X (ddof = 1), like np.cov(X.T)
                const cov: number[][] = [];
                for (let a = 0; a < d; a++) cov.push(new Array(d).fill(0));
                for (let i = 0; i < n; i++) {
                    for (let a = 0; a < d; a++) {
                        const da = X[i][a] - colMeans[a];
                        for (let b = a; b < d; b++) cov[a][b] += da * (X[i][b] - colMeans[b]);
                    }
                }
                for (let a = 0; a < d; a++) {
                    for (let b = a; b < d; b++) {
                        cov[a][b] /= ddof;
                        cov[b][a] = cov[a][b];
                    }
                }
                covPrior = cov;
            } else {
                const variances = new Array(d).fill(0);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < d; j++) {
                        const diff = X[i][j] - colMeans[j];
                        variances[j] += diff * diff;
                    }
                }
                for (let j = 0; j < d; j++) variances[j] /= ddof;
                covPrior = variances;
            }
        }
        return {
            wcPrior: this.weightConcentrationPrior ?? 1 / this.nComponents,
            mpPrior: this.meanPrecisionPrior ?? 1,
            meanPrior: this.meanPrior ? this.meanPrior.slice() : colMeans,
            dofPrior,
            covPrior,
        };
    }

    // -----------------------------------------------------------------------
    // M-step (variational updates)
    // -----------------------------------------------------------------------

    private mStep(X: number[][], resp: number[][], priors: VBPriors): VBState {
        const d = X[0].length;
        const k = this.nComponents;
        const { nk, means: xk, covariances: sk } = estimateGaussianParameters(X, resp, this.regCovar, this.covarianceType);

        // ---- weight concentration update ----
        let wcA: number[];
        let wcB: number[] | null;
        if (this.weightConcentrationPriorType === 'dirichletProcess') {
            // stick-breaking Beta posteriors: aₖ = 1 + nₖ, bₖ = γ₀ + Σ_{l>k} n_l
            wcA = nk.map((v) => 1 + v);
            wcB = new Array(k).fill(0);
            let tail = 0;
            for (let c = k - 1; c >= 0; c--) {
                wcB[c] = priors.wcPrior + tail;
                tail += nk[c];
            }
        } else {
            wcA = nk.map((v) => priors.wcPrior + v);
            wcB = null;
        }

        // ---- Gaussian mean updates ----
        const meanPrecision = nk.map((v) => priors.mpPrior + v);
        const means: number[][] = [];
        for (let c = 0; c < k; c++) {
            const row = new Array(d).fill(0);
            for (let j = 0; j < d; j++) {
                row[j] = (priors.mpPrior * priors.meanPrior[j] + nk[c] * xk[c][j]) / meanPrecision[c];
            }
            means.push(row);
        }

        // ---- Wishart precision updates (stored as normalized covariances) ----
        const dof = nk.map((v) => priors.dofPrior + v);
        let covariances: BayesianCovariances;
        if (this.covarianceType === 'full') {
            const covPrior = priors.covPrior as number[][];
            const skFull = sk as number[][][];
            const covs: number[][][] = [];
            for (let c = 0; c < k; c++) {
                const cov: number[][] = [];
                const factor = (nk[c] * priors.mpPrior) / meanPrecision[c];
                for (let a = 0; a < d; a++) {
                    const row = new Array(d).fill(0);
                    const da = xk[c][a] - priors.meanPrior[a];
                    for (let b = 0; b < d; b++) {
                        const db = xk[c][b] - priors.meanPrior[b];
                        row[b] = (covPrior[a][b] + nk[c] * skFull[c][a][b] + factor * da * db) / dof[c];
                    }
                    cov.push(row);
                }
                covs.push(cov);
            }
            covariances = covs;
        } else {
            const covPrior = priors.covPrior as number[];
            const skDiag = sk as number[][];
            const covs: number[][] = [];
            for (let c = 0; c < k; c++) {
                const row = new Array(d).fill(0);
                const factor = priors.mpPrior / meanPrecision[c];
                for (let j = 0; j < d; j++) {
                    const diff = xk[c][j] - priors.meanPrior[j];
                    row[j] = (covPrior[j] + nk[c] * (skDiag[c][j] + factor * diff * diff)) / dof[c];
                }
                covs.push(row);
            }
            covariances = covs;
        }
        return { wcA, wcB, meanPrecision, means, dof, covariances };
    }

    // -----------------------------------------------------------------------
    // E-step (expected log-likelihood)
    // -----------------------------------------------------------------------

    /** E[log πₖ] under the Dirichlet / stick-breaking posterior */
    private static expectedLogWeights(wcA: number[], wcB: number[] | null): number[] {
        const k = wcA.length;
        const out = new Array(k).fill(0);
        if (wcB !== null) {
            let cumulative = 0;
            for (let c = 0; c < k; c++) {
                const dgSum = digamma(wcA[c] + wcB[c]);
                out[c] = digamma(wcA[c]) - dgSum + cumulative;
                cumulative += digamma(wcB[c]) - dgSum;
            }
        } else {
            const total = wcA.reduce((s, v) => s + v, 0);
            const dgTotal = digamma(total);
            for (let c = 0; c < k; c++) out[c] = digamma(wcA[c]) - dgTotal;
        }
        return out;
    }

    /** n × k matrix E[log πₖ] + E[log N(xᵢ | μₖ, Λₖ⁻¹)] */
    private weightedLogProbFromState(X: number[][], state: VBState): number[][] {
        const d = X[0].length;
        const k = this.nComponents;
        // covariances_ are Wishart-normalized (divided by dof), so remove the
        // 0.5·d·log(dof) the Gaussian log-det implicitly added
        const logGauss = estimateLogGaussianProb(X, state.means, state.covariances, this.covarianceType);
        const logLambda = new Array(k).fill(0);
        for (let c = 0; c < k; c++) {
            let s = d * Math.LN2;
            for (let j = 0; j < d; j++) s += digamma(0.5 * (state.dof[c] - j));
            logLambda[c] = s;
        }
        const logWeights = BayesianGaussianMixture.expectedLogWeights(state.wcA, state.wcB);
        for (let i = 0; i < X.length; i++) {
            for (let c = 0; c < k; c++) {
                logGauss[i][c] +=
                    -0.5 * d * Math.log(state.dof[c]) +
                    0.5 * (logLambda[c] - d / state.meanPrecision[c]) +
                    logWeights[c];
            }
        }
        return logGauss;
    }

    // -----------------------------------------------------------------------
    // lower bound (sklearn's simplified ELBO, constant terms dropped)
    // -----------------------------------------------------------------------

    private computeLowerBound(logResp: number[][], state: VBState, nFeatures: number): number {
        const d = nFeatures;
        const k = this.nComponents;
        const ldpc = computeLogDetCholesky(state.covariances, this.covarianceType, k, d).map(
            (v, c) => v - 0.5 * d * Math.log(state.dof[c])
        );
        let logWishart = 0;
        for (let c = 0; c < k; c++) {
            let sumGamma = 0;
            for (let j = 0; j < d; j++) sumGamma += logGamma(0.5 * (state.dof[c] - j));
            logWishart += -(state.dof[c] * ldpc[c] + state.dof[c] * d * 0.5 * Math.LN2 + sumGamma);
        }
        let logNormWeight: number;
        if (state.wcB !== null) {
            let s = 0;
            for (let c = 0; c < k; c++) s += betaln(state.wcA[c], state.wcB[c]);
            logNormWeight = -s;
        } else {
            // note: sklearn negates the Beta normalizations (DP branch above) but
            // uses the Dirichlet normalization log Γ(Σα) − Σ log Γ(αₖ) as-is
            const total = state.wcA.reduce((s, v) => s + v, 0);
            let sumGamma = 0;
            for (let c = 0; c < k; c++) sumGamma += logGamma(state.wcA[c]);
            logNormWeight = logGamma(total) - sumGamma;
        }
        let entropy = 0; // −Σ resp·log resp
        for (let i = 0; i < logResp.length; i++) {
            for (let c = 0; c < k; c++) {
                const lr = logResp[i][c];
                const r = Math.exp(lr);
                if (r > 0) entropy -= r * lr;
            }
        }
        let sumLogMeanPrecision = 0;
        for (let c = 0; c < k; c++) sumLogMeanPrecision += Math.log(state.meanPrecision[c]);
        return entropy - logWishart - logNormWeight - 0.5 * d * sumLogMeanPrecision;
    }

    // -----------------------------------------------------------------------
    // fit
    // -----------------------------------------------------------------------

    private runVB(X: number[][], priors: VBPriors, rng: () => number): VBRunResult {
        const n = X.length;
        const d = X[0].length;
        const k = this.nComponents;
        const initResp = initializeResponsibilities(X, k, this.initParams, rng);
        let state = this.mStep(X, initResp, priors);
        let lowerBound = -Infinity;
        let converged = false;
        let nIter = 0;
        const history: number[] = [];
        for (let iter = 1; iter <= this.maxIter; iter++) {
            const prevLowerBound = lowerBound;
            // ---- E-step ----
            const weighted = this.weightedLogProbFromState(X, state);
            const logResp: number[][] = [];
            const resp: number[][] = [];
            for (let i = 0; i < n; i++) {
                const norm = logsumexp(weighted[i]);
                const lrRow = new Array(k).fill(0);
                const rRow = new Array(k).fill(0);
                for (let c = 0; c < k; c++) {
                    lrRow[c] = weighted[i][c] - norm;
                    rRow[c] = Math.exp(lrRow[c]);
                }
                logResp.push(lrRow);
                resp.push(rRow);
            }
            // ---- M-step ----
            state = this.mStep(X, resp, priors);
            lowerBound = this.computeLowerBound(logResp, state, d);
            history.push(lowerBound);
            nIter = iter;
            if (prevLowerBound !== -Infinity && Math.abs(lowerBound - prevLowerBound) < this.tol) {
                converged = true;
                break;
            }
        }
        return { ...state, lowerBound, lowerBoundHistory: history, converged, nIter };
    }

    public fit(X: number[][]): this {
        if (!X || X.length === 0) throw new Error('BayesianGaussianMixture.fit requires a non-empty X');
        if (X.length < this.nComponents) {
            throw new Error(`Expected n_samples >= nComponents but got nComponents = ${this.nComponents}, n_samples = ${X.length}`);
        }
        const priors = this.resolvePriors(X);
        const rng = createRandomGenerator(this.randomState);
        let best: VBRunResult | null = null;
        for (let run = 0; run < this.nInit; run++) {
            const result = this.runVB(X, priors, rng);
            if (best === null || result.lowerBound > best.lowerBound) best = result;
        }
        this.weightConcentrationA_ = best!.wcA;
        this.weightConcentrationB_ = best!.wcB;
        this.meanPrecision_ = best!.meanPrecision;
        this.means_ = best!.means;
        this.degreesOfFreedom_ = best!.dof;
        this.covariances_ = best!.covariances;
        this.converged = best!.converged;
        this.nIter = best!.nIter;
        this.lowerBound_ = best!.lowerBound;
        this.lowerBoundHistory_ = best!.lowerBoundHistory;
        this.weights_ = BayesianGaussianMixture.expectedWeights(best!.wcA, best!.wcB);
        return this;
    }

    /** posterior expected mixture weights (normalized) */
    private static expectedWeights(wcA: number[], wcB: number[] | null): number[] {
        const k = wcA.length;
        if (wcB !== null) {
            const weights = new Array(k).fill(0);
            let stick = 1;
            for (let c = 0; c < k; c++) {
                const total = wcA[c] + wcB[c];
                weights[c] = (wcA[c] / total) * stick;
                stick *= wcB[c] / total;
            }
            const sum = weights.reduce((s, v) => s + v, 0);
            return weights.map((v) => v / sum);
        }
        const total = wcA.reduce((s, v) => s + v, 0);
        return wcA.map((v) => v / total);
    }

    private checkFitted(): void {
        if (this.means_ === null) {
            throw new Error('This BayesianGaussianMixture instance is not fitted yet. Call fit(X) first.');
        }
    }

    private fittedState(): VBState {
        this.checkFitted();
        return {
            wcA: this.weightConcentrationA_!,
            wcB: this.weightConcentrationB_,
            meanPrecision: this.meanPrecision_!,
            means: this.means_!,
            dof: this.degreesOfFreedom_!,
            covariances: this.covariances_!,
        };
    }

    // -----------------------------------------------------------------------
    // inference API
    // -----------------------------------------------------------------------

    /** hard assignment: argmax of the responsibilities */
    public predict(X: number[][]): number[] {
        const weighted = this.weightedLogProbFromState(X, this.fittedState());
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
        const weighted = this.weightedLogProbFromState(X, this.fittedState());
        return weighted.map((row) => {
            const norm = logsumexp(row);
            return row.map((v) => Math.exp(v - norm));
        });
    }

    public fitPredict(X: number[][]): number[] {
        this.fit(X);
        return this.predict(X);
    }

    /** per-sample expected log-likelihood under the variational posterior */
    public scoreSamples(X: number[][]): number[] {
        const weighted = this.weightedLogProbFromState(X, this.fittedState());
        return weighted.map((row) => logsumexp(row));
    }

    /** average per-sample expected log-likelihood */
    public score(X: number[][]): number {
        const scores = this.scoreSamples(X);
        return scores.reduce((s, v) => s + v, 0) / scores.length;
    }

    public getWeights(): number[] | null {
        return this.weights_;
    }

    public getMeans(): number[][] | null {
        return this.means_;
    }

    public getCovariances(): BayesianCovariances | null {
        return this.covariances_;
    }

    /** final variational lower bound (sklearn's simplified ELBO) of the best run */
    public getLowerBound(): number {
        return this.lowerBound_;
    }

    /** lower-bound trajectory of the best run (one entry per iteration, non-decreasing) */
    public getLowerBoundHistory(): number[] {
        return this.lowerBoundHistory_.slice();
    }
}
registerEstimator('BayesianGaussianMixture', BayesianGaussianMixture);
