import { BaseEstimator, RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { lstsq } from '../algebra/lstsq';
import { createRandomGenerator } from '../utils/random';
import { dot, solveLinear, validateRegressionData } from '../utils/numerics';
import { LinearRegression } from './linearRegression';

function median(values: number[]): number { const sorted = values.slice().sort((a, b) => a - b), middle = (sorted.length - 1) / 2; return (sorted[Math.floor(middle)] + sorted[Math.ceil(middle)]) / 2; }
function augmented(X: number[][], fitIntercept: boolean): number[][] { return fitIntercept ? X.map(row => [1, ...row]) : X.map(row => row.slice()); }
function ridgeLeastSquares(X: number[][], y: number[], weights: number[], alpha: number, fitIntercept: boolean): number[] {
    const design = augmented(X, fitIntercept), rows = design.map((row, i) => row.map(value => value * Math.sqrt(weights[i]))), target = y.map((value, i) => value * Math.sqrt(weights[i]));
    for (let j = fitIntercept ? 1 : 0; j < design[0].length; j++) { const row = new Array(design[0].length).fill(0); row[j] = Math.sqrt(alpha); rows.push(row); target.push(0); }
    const result = lstsq(rows, target); if (result === false) throw new Error('weighted regression system is singular'); return result;
}

export interface HuberRegressorProps { epsilon?: number; maxIter?: number; alpha?: number; tol?: number; fitIntercept?: boolean; }
function huberObjective(parameters: number[], X: number[][], y: number[], epsilon: number, alpha: number, fitIntercept: boolean): { loss: number; gradient: number[] } {
    const p = X[0].length, interceptIndex = fitIntercept ? p : -1, sigmaIndex = fitIntercept ? p + 1 : p;
    const sigma = Math.max(parameters[sigmaIndex], 1e-12), gradient = new Array(parameters.length).fill(0);
    let squaredLoss = 0, outlierLoss = 0, outliers = 0, interceptGradient = 0;
    for (let i = 0; i < X.length; i++) {
        const residual = y[i] - dot(X[i], parameters) - (fitIntercept ? parameters[interceptIndex] : 0), absolute = Math.abs(residual);
        if (absolute > epsilon * sigma) {
            const sign = residual < 0 ? -1 : 1; outliers++; outlierLoss += 2 * epsilon * absolute - sigma * epsilon ** 2;
            for (let j = 0; j < p; j++) gradient[j] -= 2 * epsilon * sign * X[i][j];
            interceptGradient -= 2 * epsilon * sign;
        } else {
            squaredLoss += residual * residual / sigma;
            for (let j = 0; j < p; j++) gradient[j] -= 2 * residual * X[i][j] / sigma;
            interceptGradient -= 2 * residual / sigma;
        }
    }
    let penalty = 0; for (let j = 0; j < p; j++) { penalty += alpha * parameters[j] ** 2; gradient[j] += 2 * alpha * parameters[j]; }
    if (fitIntercept) gradient[interceptIndex] = interceptGradient;
    gradient[sigmaIndex] = X.length - outliers * epsilon ** 2 - squaredLoss / sigma;
    return { loss: X.length * sigma + squaredLoss + outlierLoss + penalty, gradient };
}
export class HuberRegressor extends RegressorBase {
    private epsilon: number; private maxIter: number; private alpha: number; private tol: number; private fitIntercept: boolean; private coefState: number[] = []; private interceptState = 0; private scaleState = 1; private outliersState: boolean[] = []; private nIterState = 0;
    constructor(props: HuberRegressorProps = {}) { super(); const { epsilon = 1.35, maxIter = 100, alpha = .0001, tol = 1e-5, fitIntercept = true } = props; if (!Number.isFinite(epsilon) || epsilon < 1 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(alpha) || alpha < 0 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid HuberRegressor parameters'); this.epsilon = epsilon; this.maxIter = maxIter; this.alpha = alpha; this.tol = tol; this.fitIntercept = fitIntercept; }
    public getParams(): Params { return { epsilon: this.epsilon, maxIter: this.maxIter, alpha: this.alpha, tol: this.tol, fitIntercept: this.fitIntercept }; }
    public fit(X: number[][], y: number[]): void {
        const p = validateRegressionData(X, y), size = p + (this.fitIntercept ? 2 : 1), sigmaIndex = size - 1, interceptIndex = this.fitIntercept ? p : -1;
        let parameters = new Array(size).fill(0); parameters[sigmaIndex] = 1;
        let inverseHessian: number[][] = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => i === j ? 1 : 0));
        let current = huberObjective(parameters, X, y, this.epsilon, this.alpha, this.fitIntercept);
        for (this.nIterState = 0; this.nIterState < this.maxIter; this.nIterState++) {
            if (Math.max(...current.gradient.map(Math.abs)) <= this.tol) break;
            let direction = inverseHessian.map(row => -dot(row, current.gradient));
            let directionalDerivative = dot(direction, current.gradient);
            if (!(directionalDerivative < 0) || !direction.every(Number.isFinite)) { direction = current.gradient.map(value => -value); directionalDerivative = -dot(current.gradient, current.gradient); inverseHessian = inverseHessian.map((row, i) => row.map((_, j) => i === j ? 1 : 0)); }
            let step = 1, nextParameters = parameters.slice(), next = current;
            while (step >= 1e-12) {
                nextParameters = parameters.map((value, i) => value + step * direction[i]); nextParameters[sigmaIndex] = Math.max(nextParameters[sigmaIndex], 1e-12);
                next = huberObjective(nextParameters, X, y, this.epsilon, this.alpha, this.fitIntercept);
                if (Number.isFinite(next.loss) && next.loss <= current.loss + 1e-4 * step * directionalDerivative) break;
                step *= .5;
            }
            if (step < 1e-12) break;
            const s = nextParameters.map((value, i) => value - parameters[i]), deltaGradient = next.gradient.map((value, i) => value - current.gradient[i]), curvature = dot(s, deltaGradient);
            if (curvature > 1e-12) {
                const rho = 1 / curvature, identityMinusSY = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0) - rho * s[i] * deltaGradient[j])), identityMinusYS = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0) - rho * deltaGradient[i] * s[j]));
                inverseHessian = identityMinusSY.map(row => identityMinusYS[0].map((_, j) => row.reduce((sum, value, k) => sum + value * inverseHessian[k][j], 0)));
                inverseHessian = inverseHessian.map(row => identityMinusYS[0].map((_, j) => row.reduce((sum, value, k) => sum + value * identityMinusYS[k][j], 0)));
                inverseHessian = inverseHessian.map((row, i) => row.map((value, j) => value + rho * s[i] * s[j]));
            }
            parameters = nextParameters; current = next;
        }
        this.interceptState = this.fitIntercept ? parameters[interceptIndex] : 0; this.coefState = parameters.slice(0, p); this.scaleState = parameters[sigmaIndex]; this.outliersState = y.map((value, i) => Math.abs(value - this.predict([X[i]])[0]) > this.epsilon * this.scaleState);
    }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('HuberRegressor is not fitted'); return X.map(row => this.interceptState + dot(row, this.coefState)); }
    public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get scale(): number { return this.scaleState; } public get outliers(): boolean[] { return this.outliersState.slice(); } public get nIter(): number { return this.nIterState; }
}
registerEstimator('HuberRegressor', HuberRegressor);

interface RegressorLike extends BaseEstimator { fit(X: number[][], y: number[]): void; predict(X: number[][]): number[]; }
export interface RANSACRegressorProps { estimator?: RegressorLike; minSamples?: number; residualThreshold?: number | null; maxTrials?: number; stopProbability?: number; randomState?: number; }
export class RANSACRegressor extends RegressorBase {
    private estimator: RegressorLike; private minSamples?: number; private residualThreshold: number | null; private maxTrials: number; private stopProbability: number; private randomState?: number; private estimatorState?: RegressorLike; private inlierMaskState: boolean[] = []; private nTrialsState = 0;
    constructor(props: RANSACRegressorProps = {}) { super(); const { estimator = new LinearRegression(), minSamples, residualThreshold = null, maxTrials = 100, stopProbability = .99, randomState } = props; if (!(estimator instanceof BaseEstimator) || minSamples !== undefined && (!Number.isInteger(minSamples) || minSamples < 1) || residualThreshold !== null && (!Number.isFinite(residualThreshold) || residualThreshold < 0) || !Number.isInteger(maxTrials) || maxTrials < 1 || !Number.isFinite(stopProbability) || stopProbability < 0 || stopProbability > 1) throw new Error('invalid RANSACRegressor parameters'); this.estimator = estimator; this.minSamples = minSamples; this.residualThreshold = residualThreshold; this.maxTrials = maxTrials; this.stopProbability = stopProbability; this.randomState = randomState; }
    public getParams(): Params { return { estimator: this.estimator, minSamples: this.minSamples, residualThreshold: this.residualThreshold, maxTrials: this.maxTrials, stopProbability: this.stopProbability, randomState: this.randomState }; }
    public fit(X: number[][], y: number[]): void { const p = validateRegressionData(X, y), minimum = this.minSamples ?? p + 1; if (minimum > X.length) throw new Error('minSamples exceeds number of samples'); const threshold = this.residualThreshold ?? median(y.map(value => Math.abs(value - median(y)))), random = createRandomGenerator(this.randomState); let bestMask: boolean[] = [], bestCount = 0, bestError = Infinity, dynamicTrials = this.maxTrials;
        for (this.nTrialsState = 1; this.nTrialsState <= Math.min(this.maxTrials, dynamicTrials); this.nTrialsState++) { const pool = X.map((_, i) => i); for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; } const sample = pool.slice(0, minimum), candidate = this.estimator.clone() as RegressorLike; try { candidate.fit(sample.map(i => X[i]), sample.map(i => y[i])); } catch { continue; } const residuals = candidate.predict(X).map((value, i) => Math.abs(y[i] - value)), mask = residuals.map(value => value <= threshold), count = mask.filter(Boolean).length, error = residuals.reduce((sum, value, i) => sum + (mask[i] ? value * value : 0), 0); if (count > bestCount || count === bestCount && error < bestError) { bestMask = mask; bestCount = count; bestError = error; const ratio = count / X.length; if (ratio > 0 && ratio < 1) dynamicTrials = Math.ceil(Math.log(1 - this.stopProbability) / Math.log(1 - ratio ** minimum)); else if (ratio === 1) dynamicTrials = 1; } }
        this.nTrialsState = Math.min(this.nTrialsState, this.maxTrials); if (bestCount < minimum) throw new Error('RANSAC could not find a valid consensus set'); this.estimatorState = this.estimator.clone() as RegressorLike; this.estimatorState.fit(X.filter((_, i) => bestMask[i]), y.filter((_, i) => bestMask[i])); this.inlierMaskState = bestMask; }
    public predict(X: number[][]): number[] { if (!this.estimatorState) throw new Error('RANSACRegressor is not fitted'); return this.estimatorState.predict(X); }
    public get estimatorFitted(): RegressorLike { if (!this.estimatorState) throw new Error('RANSACRegressor is not fitted'); return this.estimatorState; } public get inlierMask(): boolean[] { return this.inlierMaskState.slice(); } public get nTrials(): number { return this.nTrialsState; }
}
registerEstimator('RANSACRegressor', RANSACRegressor);

export interface TheilSenRegressorProps { fitIntercept?: boolean; maxSubpopulation?: number; nSubsamples?: number | null; maxIter?: number; tol?: number; randomState?: number; }
function combinations(n: number, k: number, limit: number, random: () => number): number[][] { const total: number[][] = []; const visit = (start: number, current: number[]) => { if (total.length > limit) return; if (current.length === k) { total.push(current.slice()); return; } for (let i = start; i <= n - (k - current.length); i++) visit(i + 1, [...current, i]); }; visit(0, []); if (total.length <= limit) return total; const sampled: number[][] = [], seen = new Set<string>(); while (sampled.length < limit) { const pool = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; } const row = pool.slice(0, k).sort((a, b) => a - b), key = row.join(','); if (!seen.has(key)) { seen.add(key); sampled.push(row); } } return sampled; }
export class TheilSenRegressor extends RegressorBase {
    private fitIntercept: boolean; private maxSubpopulation: number; private nSubsamples: number | null; private maxIter: number; private tol: number; private randomState?: number; private coefState: number[] = []; private interceptState = 0; private nIterState = 0;
    constructor(props: TheilSenRegressorProps = {}) { super(); const { fitIntercept = true, maxSubpopulation = 1e4, nSubsamples = null, maxIter = 300, tol = 1e-3, randomState } = props; if (!Number.isFinite(maxSubpopulation) || maxSubpopulation < 1 || nSubsamples !== null && (!Number.isInteger(nSubsamples) || nSubsamples < 1) || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid TheilSenRegressor parameters'); this.fitIntercept = fitIntercept; this.maxSubpopulation = Math.floor(maxSubpopulation); this.nSubsamples = nSubsamples; this.maxIter = maxIter; this.tol = tol; this.randomState = randomState; }
    public getParams(): Params { return { fitIntercept: this.fitIntercept, maxSubpopulation: this.maxSubpopulation, nSubsamples: this.nSubsamples, maxIter: this.maxIter, tol: this.tol, randomState: this.randomState }; }
    public fit(X: number[][], y: number[]): void { const p = validateRegressionData(X, y), k = this.nSubsamples ?? Math.min(X.length, p + (this.fitIntercept ? 1 : 0)); if (k > X.length || k < p + (this.fitIntercept ? 1 : 0)) throw new Error('nSubsamples must permit an identifiable linear model'); const subsets = combinations(X.length, k, this.maxSubpopulation, createRandomGenerator(this.randomState)), estimates: number[][] = []; for (const subset of subsets) { const result = lstsq(augmented(subset.map(i => X[i]), this.fitIntercept), subset.map(i => y[i])); if (result !== false && result.every(Number.isFinite)) estimates.push(result); } if (estimates.length === 0) throw new Error('TheilSen could not fit any subsample'); let center = Array.from({ length: estimates[0].length }, (_, j) => estimates.reduce((sum, row) => sum + row[j], 0) / estimates.length); for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) { const distances = estimates.map(row => Math.sqrt(row.reduce((sum, value, j) => sum + (value - center[j]) ** 2, 0))); const coincident = distances.findIndex(value => value < 1e-12); if (coincident >= 0) { center = estimates[coincident].slice(); break; } const next = center.map((_, j) => estimates.reduce((sum, row, i) => sum + row[j] / distances[i], 0) / estimates.reduce((sum, _, i) => sum + 1 / distances[i], 0)); if (Math.sqrt(next.reduce((sum, value, j) => sum + (value - center[j]) ** 2, 0)) < this.tol) { center = next; break; } center = next; } this.nIterState = Math.min(this.nIterState, this.maxIter); this.interceptState = this.fitIntercept ? center[0] : 0; this.coefState = center.slice(this.fitIntercept ? 1 : 0); }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('TheilSenRegressor is not fitted'); return X.map(row => this.interceptState + dot(row, this.coefState)); } public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get nIter(): number { return this.nIterState; }
}
registerEstimator('TheilSenRegressor', TheilSenRegressor);

export interface QuantileRegressorProps { quantile?: number; alpha?: number; fitIntercept?: boolean; maxIter?: number; tol?: number; }
export class QuantileRegressor extends RegressorBase {
    private quantile: number; private alpha: number; private fitIntercept: boolean; private maxIter: number; private tol: number; private coefState: number[] = []; private interceptState = 0; private nIterState = 0;
    constructor(props: QuantileRegressorProps = {}) { super(); const { quantile = .5, alpha = 1, fitIntercept = true, maxIter = 5000, tol = 1e-7 } = props; if (!Number.isFinite(quantile) || quantile <= 0 || quantile >= 1 || !Number.isFinite(alpha) || alpha < 0 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid QuantileRegressor parameters'); this.quantile = quantile; this.alpha = alpha; this.fitIntercept = fitIntercept; this.maxIter = maxIter; this.tol = tol; }
    public getParams(): Params { return { quantile: this.quantile, alpha: this.alpha, fitIntercept: this.fitIntercept, maxIter: this.maxIter, tol: this.tol }; }
    public fit(X: number[][], y: number[]): void { validateRegressionData(X, y); const D = augmented(X, this.fitIntercept), d = D[0].length, n = X.length, gram = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => D.reduce((sum, row) => sum + row[a] * row[b], 0) + (a === b ? 1 : 0))); let w = new Array(d).fill(0), z = new Array(d).fill(0), residual = y.slice(), u1 = new Array(n).fill(0), u2 = new Array(d).fill(0); const lambda = 1 / n;
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) { const rhs = Array.from({ length: d }, (_, j) => D.reduce((sum, row, i) => sum + row[j] * (y[i] - residual[i] - u1[i]), 0) + z[j] - u2[j]); w = solveLinear(gram, rhs); const fitted = D.map(row => dot(row, w)), previousResidual = residual.slice(), previousZ = z.slice(); residual = y.map((value, i) => { const v = value - fitted[i] - u1[i]; return v > lambda * this.quantile ? v - lambda * this.quantile : v < -lambda * (1 - this.quantile) ? v + lambda * (1 - this.quantile) : 0; }); z = w.map((value, j) => { if (this.fitIntercept && j === 0) return value + u2[j]; const v = value + u2[j], threshold = this.alpha; return Math.sign(v) * Math.max(Math.abs(v) - threshold, 0); }); let primal = 0, dual = 0; for (let i = 0; i < n; i++) { const constraint = fitted[i] + residual[i] - y[i]; u1[i] += constraint; primal += constraint * constraint; dual += (residual[i] - previousResidual[i]) ** 2; } for (let j = 0; j < d; j++) { const constraint = w[j] - z[j]; u2[j] += constraint; primal += constraint * constraint; dual += (z[j] - previousZ[j]) ** 2; } if (Math.sqrt(primal) < this.tol && Math.sqrt(dual) < this.tol) break; }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.interceptState = this.fitIntercept ? z[0] : 0; this.coefState = z.slice(this.fitIntercept ? 1 : 0); }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('QuantileRegressor is not fitted'); return X.map(row => this.interceptState + dot(row, this.coefState)); } public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get nIter(): number { return this.nIterState; }
}
registerEstimator('QuantileRegressor', QuantileRegressor);
