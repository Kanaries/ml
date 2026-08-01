import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { dot, solveLinear, validateRegressionData } from '../utils/numerics';

export type GLMLink = 'auto' | 'identity' | 'log';
export interface TweedieRegressorProps { power?: number; alpha?: number; fitIntercept?: boolean; link?: GLMLink; maxIter?: number; tol?: number; }

class BaseGLM extends RegressorBase {
    protected power: number; protected regularization: number; protected fitIntercept: boolean; protected link: GLMLink; protected maxIter: number; protected tol: number;
    protected coefState: number[] = []; protected interceptState = 0; protected nIterState = 0;
    constructor(props: TweedieRegressorProps) { super(); const { power = 0, alpha = 1, fitIntercept = true, link = 'auto', maxIter = 100, tol = 1e-4 } = props; if (!Number.isFinite(power) || power > 0 && power < 1 || !Number.isFinite(alpha) || alpha < 0 || !['auto', 'identity', 'log'].includes(link) || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid TweedieRegressor parameters'); this.power = power; this.regularization = alpha; this.fitIntercept = fitIntercept; this.link = link; this.maxIter = maxIter; this.tol = tol; }
    public getParams(): Params { return { power: this.power, alpha: this.regularization, fitIntercept: this.fitIntercept, link: this.link, maxIter: this.maxIter, tol: this.tol }; }
    private activeLink(): 'identity' | 'log' { return this.link === 'auto' ? (this.power <= 0 ? 'identity' : 'log') : this.link; }
    private mean(eta: number): number { return this.activeLink() === 'log' ? Math.exp(Math.max(-30, Math.min(30, eta))) : eta; }
    private loss(y: number[], design: number[][], weights: number[]): number {
        let total = 0; for (let i = 0; i < y.length; i++) { const mu = this.mean(dot(design[i], weights)); if (this.power > 0 && mu <= 0) return Infinity; if (this.power === 0) total += .5 * (y[i] - mu) ** 2; else if (this.power === 1) total += mu - y[i] * Math.log(mu); else if (this.power === 2) total += Math.log(mu) + y[i] / mu; else total += mu ** (2 - this.power) / (2 - this.power) - y[i] * mu ** (1 - this.power) / (1 - this.power); }
        total /= y.length; for (let j = this.fitIntercept ? 1 : 0; j < weights.length; j++) total += .5 * this.regularization * weights[j] ** 2; return total;
    }
    public fit(X: number[][], y: number[]): void {
        const p = validateRegressionData(X, y); if (this.power >= 2 && y.some(value => value <= 0) || this.power >= 1 && this.power < 2 && y.some(value => value < 0)) throw new Error('target values are outside the Tweedie domain');
        const design = this.fitIntercept ? X.map(row => [1, ...row]) : X.map(row => row.slice()), d = design[0].length, link = this.activeLink(); let weights = new Array(d).fill(0);
        if (this.fitIntercept) { const meanY = y.reduce((a, b) => a + b, 0) / y.length; weights[0] = link === 'log' ? Math.log(Math.max(meanY, 1e-12)) : meanY; }
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            const gradient = new Array(d).fill(0), hessian = Array.from({ length: d }, () => new Array(d).fill(0));
            for (let i = 0; i < y.length; i++) {
                const mu = this.mean(dot(design[i], weights)); let g: number, h: number;
                if (link === 'identity') {
                    if (this.power === 0) { g = mu - y[i]; h = 1; }
                    else {
                        const safeMu = Math.max(mu, 1e-12);
                        g = safeMu ** (1 - this.power) - y[i] * safeMu ** (-this.power);
                        h = (1 - this.power) * safeMu ** (-this.power) + this.power * y[i] * safeMu ** (-this.power - 1);
                    }
                }
                else { g = mu ** (2 - this.power) - y[i] * mu ** (1 - this.power); h = (2 - this.power) * mu ** (2 - this.power) - (1 - this.power) * y[i] * mu ** (1 - this.power); }
                h = Math.max(h, 1e-10); for (let a = 0; a < d; a++) { gradient[a] += design[i][a] * g / y.length; for (let b = 0; b < d; b++) hessian[a][b] += design[i][a] * design[i][b] * h / y.length; }
            }
            for (let j = this.fitIntercept ? 1 : 0; j < d; j++) { gradient[j] += this.regularization * weights[j]; hessian[j][j] += this.regularization; }
            let step: number[]; try { step = solveLinear(hessian, gradient); } catch { break; }
            const oldLoss = this.loss(y, design, weights); let rate = 1, candidate = weights.map((value, j) => value - rate * step[j]); while (rate > 1e-8 && this.loss(y, design, candidate) > oldLoss) { rate /= 2; candidate = weights.map((value, j) => value - rate * step[j]); }
            weights = candidate; if (Math.max(...step.map(value => Math.abs(rate * value))) < this.tol) break;
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.interceptState = this.fitIntercept ? weights[0] : 0; this.coefState = weights.slice(this.fitIntercept ? 1 : 0); if (this.coefState.length !== p) throw new Error('internal GLM shape mismatch');
    }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('GLM is not fitted'); return X.map(row => this.mean(this.interceptState + dot(row, this.coefState))); }
    public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get nIter(): number { return this.nIterState; }
}

export class TweedieRegressor extends BaseGLM { constructor(props: TweedieRegressorProps = {}) { super(props); } }
registerEstimator('TweedieRegressor', TweedieRegressor);
export interface PoissonRegressorProps extends Omit<TweedieRegressorProps, 'power' | 'link'> {}
export class PoissonRegressor extends BaseGLM { constructor(props: PoissonRegressorProps = {}) { super({ ...props, power: 1, link: 'log' }); } public getParams(): Params { const { power: _, link: __, ...params } = super.getParams(); return params; } }
registerEstimator('PoissonRegressor', PoissonRegressor);
export interface GammaRegressorProps extends Omit<TweedieRegressorProps, 'power' | 'link'> {}
export class GammaRegressor extends BaseGLM { constructor(props: GammaRegressorProps = {}) { super({ ...props, power: 2, link: 'log' }); } public getParams(): Params { const { power: _, link: __, ...params } = super.getParams(); return params; } }
registerEstimator('GammaRegressor', GammaRegressor);
