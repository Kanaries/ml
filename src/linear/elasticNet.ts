import { RegressorBase } from '../base/regressor';
import { registerEstimator, Params } from '../base/estimator';

function softThreshold(value: number, threshold: number): number {
    if (value > threshold) {
        return value - threshold;
    }
    if (value < -threshold) {
        return value + threshold;
    }
    return 0;
}

function validateRegressionData(X: number[][], Y: number[]): void {
    if (X.length === 0 || Y.length === 0) {
        throw new Error('X and Y must be non-empty');
    }
    if (X.length !== Y.length) {
        throw new Error('X and Y must have the same length');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must have at least one feature');
    }
    for (const row of X) {
        if (row.length !== nFeatures) {
            throw new Error('all rows in X must have the same length');
        }
    }
}

export interface ElasticNetProps {
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    tol?: number;
}

export class ElasticNet extends RegressorBase {
    private alpha: number;
    private l1Ratio: number;
    private fitIntercept: boolean;
    private maxIter: number;
    private tol: number;
    private coef: number[];
    private intercept: number;
    private fitted: boolean;
    private edgeModel: { predict: (X: number[][]) => number[] } | null;

    constructor(props: ElasticNetProps = {}) {
        super();
        const {
            alpha = 1,
            l1Ratio = 0.5,
            fitIntercept = true,
            maxIter = 1000,
            tol = 1e-6,
        } = props;
        if (!Number.isFinite(alpha) || alpha < 0) {
            throw new Error('alpha must be a finite number >= 0');
        }
        if (!Number.isFinite(l1Ratio) || l1Ratio < 0 || l1Ratio > 1) {
            throw new Error('l1Ratio must be a finite number between 0 and 1');
        }
        if (!Number.isInteger(maxIter) || maxIter <= 0) {
            throw new Error('maxIter must be an integer > 0');
        }
        if (!Number.isFinite(tol) || tol <= 0) {
            throw new Error('tol must be a finite number > 0');
        }
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.coef = [];
        this.intercept = 0;
        this.fitted = false;
        this.edgeModel = null;
    }

    public getParams(): Params {
        return {
            alpha: this.alpha,
            l1Ratio: this.l1Ratio,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tol: this.tol,
        };
    }

    public fit(X: number[][], Y: number[]): void {
        validateRegressionData(X, Y);
        this.edgeModel = null;

        if (this.l1Ratio === 0) {
            // sklearn ElasticNet minimizes 1/(2n)||y-Xw||^2 + 0.5*alpha*||w||^2,
            // while Ridge minimizes ||y-Xw||^2 + alpha*||w||^2. The residual term
            // differs by a factor n, so ElasticNet(alpha, l1_ratio=0) == Ridge(n*alpha).
            const ridge = new RidgeRegression({ alpha: this.alpha * X.length, fitIntercept: this.fitIntercept });
            ridge.fit(X, Y);
            this.edgeModel = ridge;
            this.fitted = true;
            return;
        }

        if (this.l1Ratio === 1) {
            const lasso = new LassoRegression({
                alpha: this.alpha,
                fitIntercept: this.fitIntercept,
                maxIter: this.maxIter,
                tol: this.tol,
            });
            lasso.fit(X, Y);
            this.edgeModel = lasso;
            this.fitted = true;
            return;
        }

        const nSamples = X.length;
        const nFeatures = X[0].length;
        const xMeans = new Array(nFeatures).fill(0);
        if (this.fitIntercept) {
            for (let j = 0; j < nFeatures; j++) {
                let sum = 0;
                for (let i = 0; i < nSamples; i++) {
                    sum += X[i][j];
                }
                xMeans[j] = sum / nSamples;
            }
        }

        const yMean = this.fitIntercept ? Y.reduce((sum, value) => sum + value, 0) / nSamples : 0;
        const centeredX = X.map(row => row.map((value, j) => value - xMeans[j]));
        const centeredY = Y.map(value => value - yMean);

        const coef = new Array(nFeatures).fill(0);
        const yPred = new Array(nSamples).fill(0);
        const z = new Array(nFeatures).fill(0);
        for (let j = 0; j < nFeatures; j++) {
            let norm = 0;
            for (let i = 0; i < nSamples; i++) {
                const xij = centeredX[i][j];
                norm += xij * xij;
            }
            z[j] = norm / nSamples;
        }

        const l1Penalty = this.alpha * this.l1Ratio;
        const l2Penalty = this.alpha * (1 - this.l1Ratio);

        for (let iter = 0; iter < this.maxIter; iter++) {
            let maxUpdate = 0;
            for (let j = 0; j < nFeatures; j++) {
                const oldCoef = coef[j];
                let rho = 0;
                for (let i = 0; i < nSamples; i++) {
                    const xij = centeredX[i][j];
                    const residual = centeredY[i] - yPred[i] + xij * oldCoef;
                    rho += xij * residual;
                }
                rho /= nSamples;

                const denom = z[j] + l2Penalty;
                const newCoef = denom === 0 ? 0 : softThreshold(rho, l1Penalty) / denom;
                const delta = newCoef - oldCoef;
                if (delta !== 0) {
                    for (let i = 0; i < nSamples; i++) {
                        yPred[i] += centeredX[i][j] * delta;
                    }
                }
                coef[j] = newCoef;
                maxUpdate = Math.max(maxUpdate, Math.abs(delta));
            }
            if (maxUpdate < this.tol) {
                break;
            }
        }

        this.coef = coef;
        this.intercept = this.fitIntercept
            ? yMean - xMeans.reduce((sum, mean, idx) => sum + mean * this.coef[idx], 0)
            : 0;
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        if (this.edgeModel) {
            return this.edgeModel.predict(X);
        }
        return X.map(row => {
            if (row.length !== this.coef.length) {
                throw new Error('input feature size does not match fitted model');
            }
            let sum = this.intercept;
            for (let i = 0; i < this.coef.length; i++) {
                sum += this.coef[i] * row[i];
            }
            return sum;
        });
    }
}
registerEstimator('ElasticNet', ElasticNet);
import { LassoRegression } from './lassoRegression';
import { RidgeRegression } from './ridgeRegression';
