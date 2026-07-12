/**
 * Machinery shared by GaussianMixture and BayesianGaussianMixture: sufficient
 * statistics (weighted means/covariances per covariance type), log Gaussian
 * densities computed through Cholesky factors, and responsibility
 * initialization (k-means / random), all mirroring
 * sklearn.mixture._gaussian_mixture.
 */
import { KMeans } from '../clusters/kmeans';
import { EPS, cholesky, solveLowerTriangular } from './math';

export type CovarianceType = 'full' | 'tied' | 'diag' | 'spherical';

/**
 * Covariance container per covariance type:
 *  - 'full':      k × d × d  (number[][][])
 *  - 'tied':      d × d      (number[][])
 *  - 'diag':      k × d      (number[][])
 *  - 'spherical': k          (number[])
 */
export type Covariances = number[][][] | number[][] | number[];

export interface GaussianSufficientStats {
    /** effective number of samples per component (Σᵢ respᵢₖ + 10·eps) */
    nk: number[];
    /** k × d weighted means */
    means: number[][];
    /** per-type covariance container (regCovar already added to diagonals) */
    covariances: Covariances;
}

/**
 * Weighted Gaussian parameter estimates from responsibilities — the M-step of
 * EM (and the sufficient statistics of variational Bayes).
 */
export function estimateGaussianParameters(
    X: number[][],
    resp: number[][],
    regCovar: number,
    covarianceType: CovarianceType
): GaussianSufficientStats {
    const n = X.length;
    const d = X[0].length;
    const k = resp[0].length;
    const nk: number[] = new Array(k).fill(10 * EPS);
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) nk[c] += resp[i][c];
    }
    const means: number[][] = [];
    for (let c = 0; c < k; c++) means.push(new Array(d).fill(0));
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
            const r = resp[i][c];
            if (r === 0) continue;
            for (let j = 0; j < d; j++) means[c][j] += r * X[i][j];
        }
    }
    for (let c = 0; c < k; c++) {
        for (let j = 0; j < d; j++) means[c][j] /= nk[c];
    }

    let covariances: Covariances;
    if (covarianceType === 'full') {
        const covs: number[][][] = [];
        for (let c = 0; c < k; c++) {
            const cov: number[][] = [];
            for (let a = 0; a < d; a++) cov.push(new Array(d).fill(0));
            for (let i = 0; i < n; i++) {
                const r = resp[i][c];
                if (r === 0) continue;
                for (let a = 0; a < d; a++) {
                    const da = X[i][a] - means[c][a];
                    for (let b = a; b < d; b++) {
                        cov[a][b] += r * da * (X[i][b] - means[c][b]);
                    }
                }
            }
            for (let a = 0; a < d; a++) {
                for (let b = a; b < d; b++) {
                    cov[a][b] /= nk[c];
                    cov[b][a] = cov[a][b];
                }
                cov[a][a] += regCovar;
            }
            covs.push(cov);
        }
        covariances = covs;
    } else if (covarianceType === 'tied') {
        // (XᵀX − Σₖ nₖ μₖμₖᵀ) / Σₖ nₖ + regCovar·I
        const cov: number[][] = [];
        for (let a = 0; a < d; a++) cov.push(new Array(d).fill(0));
        for (let i = 0; i < n; i++) {
            for (let a = 0; a < d; a++) {
                for (let b = a; b < d; b++) cov[a][b] += X[i][a] * X[i][b];
            }
        }
        let nkSum = 0;
        for (let c = 0; c < k; c++) nkSum += nk[c];
        for (let c = 0; c < k; c++) {
            for (let a = 0; a < d; a++) {
                for (let b = a; b < d; b++) cov[a][b] -= nk[c] * means[c][a] * means[c][b];
            }
        }
        for (let a = 0; a < d; a++) {
            for (let b = a; b < d; b++) {
                cov[a][b] /= nkSum;
                cov[b][a] = cov[a][b];
            }
            cov[a][a] += regCovar;
        }
        covariances = cov;
    } else {
        // 'diag' and 'spherical' both start from per-feature variances
        const diag: number[][] = [];
        for (let c = 0; c < k; c++) diag.push(new Array(d).fill(0));
        for (let i = 0; i < n; i++) {
            for (let c = 0; c < k; c++) {
                const r = resp[i][c];
                if (r === 0) continue;
                for (let j = 0; j < d; j++) {
                    const diff = X[i][j] - means[c][j];
                    diag[c][j] += r * diff * diff;
                }
            }
        }
        for (let c = 0; c < k; c++) {
            for (let j = 0; j < d; j++) diag[c][j] = diag[c][j] / nk[c] + regCovar;
        }
        if (covarianceType === 'diag') {
            covariances = diag;
        } else {
            covariances = diag.map((row) => row.reduce((s, v) => s + v, 0) / d);
        }
    }
    return { nk, means, covariances };
}

/**
 * n × k matrix of log N(xᵢ | μₖ, Σₖ). 'full'/'tied' go through a Cholesky
 * factor (log-det from the diagonal, quadratic form by triangular solve);
 * 'diag'/'spherical' use closed forms.
 */
export function estimateLogGaussianProb(
    X: number[][],
    means: number[][],
    covariances: Covariances,
    covarianceType: CovarianceType
): number[][] {
    const n = X.length;
    const k = means.length;
    const d = X[0].length;
    const cst = d * Math.log(2 * Math.PI);
    const out: number[][] = [];
    for (let i = 0; i < n; i++) out.push(new Array(k).fill(0));

    if (covarianceType === 'full' || covarianceType === 'tied') {
        const tiedL = covarianceType === 'tied' ? cholesky(covariances as number[][]) : null;
        for (let c = 0; c < k; c++) {
            const L = tiedL ?? cholesky((covariances as number[][][])[c]);
            let logDet = 0;
            for (let a = 0; a < d; a++) logDet += Math.log(L[a][a]);
            logDet *= 2;
            const diff = new Array(d).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < d; j++) diff[j] = X[i][j] - means[c][j];
                const z = solveLowerTriangular(L, diff);
                let quad = 0;
                for (let j = 0; j < d; j++) quad += z[j] * z[j];
                out[i][c] = -0.5 * (cst + logDet + quad);
            }
        }
    } else if (covarianceType === 'diag') {
        const covs = covariances as number[][];
        for (let c = 0; c < k; c++) {
            let logDet = 0;
            for (let j = 0; j < d; j++) logDet += Math.log(covs[c][j]);
            for (let i = 0; i < n; i++) {
                let quad = 0;
                for (let j = 0; j < d; j++) {
                    const diff = X[i][j] - means[c][j];
                    quad += (diff * diff) / covs[c][j];
                }
                out[i][c] = -0.5 * (cst + logDet + quad);
            }
        }
    } else {
        const covs = covariances as number[];
        for (let c = 0; c < k; c++) {
            const logDet = d * Math.log(covs[c]);
            for (let i = 0; i < n; i++) {
                let quad = 0;
                for (let j = 0; j < d; j++) {
                    const diff = X[i][j] - means[c][j];
                    quad += diff * diff;
                }
                out[i][c] = -0.5 * (cst + logDet + quad / covs[c]);
            }
        }
    }
    return out;
}

/**
 * Per-component log-determinant of the precision Cholesky factor,
 * i.e. −½·log det Σₖ (sklearn's `_compute_log_det_cholesky`). 'tied' is
 * replicated across the k components.
 */
export function computeLogDetCholesky(
    covariances: Covariances,
    covarianceType: CovarianceType,
    nComponents: number,
    nFeatures: number
): number[] {
    const out: number[] = new Array(nComponents).fill(0);
    if (covarianceType === 'full') {
        for (let c = 0; c < nComponents; c++) {
            const L = cholesky((covariances as number[][][])[c]);
            let v = 0;
            for (let a = 0; a < nFeatures; a++) v -= Math.log(L[a][a]);
            out[c] = v;
        }
    } else if (covarianceType === 'tied') {
        const L = cholesky(covariances as number[][]);
        let v = 0;
        for (let a = 0; a < nFeatures; a++) v -= Math.log(L[a][a]);
        out.fill(v);
    } else if (covarianceType === 'diag') {
        for (let c = 0; c < nComponents; c++) {
            let v = 0;
            for (let j = 0; j < nFeatures; j++) v -= 0.5 * Math.log((covariances as number[][])[c][j]);
            out[c] = v;
        }
    } else {
        for (let c = 0; c < nComponents; c++) {
            out[c] = -0.5 * nFeatures * Math.log((covariances as number[])[c]);
        }
    }
    return out;
}

/**
 * Initial responsibilities: one-hot k-means labels or normalized uniform
 * random rows. The k-means run is seeded from the fit-local RNG so nInit
 * restarts differ but a seeded fit stays fully deterministic.
 */
export function initializeResponsibilities(
    X: number[][],
    nComponents: number,
    initParams: 'kmeans' | 'random',
    rng: () => number
): number[][] {
    const n = X.length;
    const resp: number[][] = [];
    if (initParams === 'kmeans') {
        const seed = Math.floor(rng() * 2147483646) + 1;
        const km = new KMeans({ n_clusters: nComponents, random_state: seed, n_init: 1, max_iter: 100 });
        const labels = km.fitPredict(X);
        for (let i = 0; i < n; i++) {
            const row = new Array(nComponents).fill(0);
            row[labels[i]] = 1;
            resp.push(row);
        }
    } else {
        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            let sum = 0;
            for (let c = 0; c < nComponents; c++) {
                // strictly positive so every row normalizes cleanly
                const v = rng() + EPS;
                row.push(v);
                sum += v;
            }
            for (let c = 0; c < nComponents; c++) row[c] /= sum;
            resp.push(row);
        }
    }
    return resp;
}
