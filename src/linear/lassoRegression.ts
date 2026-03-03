function softThreshold(value: number, threshold: number): number {
    if (value > threshold) {
        return value - threshold;
    }
    if (value < -threshold) {
        return value + threshold;
    }
    return 0;
}

export interface LassoRegressionProps {
    alpha?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    tol?: number;
}

export class LassoRegression {
    private alpha: number;
    private fitIntercept: boolean;
    private maxIter: number;
    private tol: number;
    private coef: number[];
    private intercept: number;
    private fitted: boolean;

    public constructor(props: LassoRegressionProps = {}) {
        const { alpha = 1, fitIntercept = true, maxIter = 1000, tol = 1e-6 } = props;
        if (!Number.isFinite(alpha) || alpha < 0) {
            throw new Error('alpha must be a finite number >= 0');
        }
        if (!Number.isInteger(maxIter) || maxIter <= 0) {
            throw new Error('maxIter must be an integer > 0');
        }
        if (!Number.isFinite(tol) || tol <= 0) {
            throw new Error('tol must be a finite number > 0');
        }
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
        this.maxIter = maxIter;
        this.tol = tol;
        this.coef = [];
        this.intercept = 0;
        this.fitted = false;
    }

    public fit(X: number[][], Y: number[]): void {
        if (X.length === 0 || Y.length === 0) {
            throw new Error('X and Y must be non-empty');
        }
        if (X.length !== Y.length) {
            throw new Error('X and Y must have the same length');
        }

        const nSamples = X.length;
        const nFeatures = X[0].length;
        if (nFeatures === 0) {
            throw new Error('X must have at least one feature');
        }
        for (const row of X) {
            if (row.length !== nFeatures) {
                throw new Error('all rows in X must have the same length');
            }
        }

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

        const yMean = this.fitIntercept ? Y.reduce((s, v) => s + v, 0) / nSamples : 0;

        const centeredX: number[][] = X.map(row => row.map((v, j) => v - xMeans[j]));
        const centeredY: number[] = Y.map(v => v - yMean);

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

        for (let iter = 0; iter < this.maxIter; iter++) {
            let maxUpdate = 0;
            for (let j = 0; j < nFeatures; j++) {
                const zj = z[j];
                const oldCoef = coef[j];
                if (zj === 0) {
                    coef[j] = 0;
                    continue;
                }

                let rho = 0;
                for (let i = 0; i < nSamples; i++) {
                    const xij = centeredX[i][j];
                    const residual = centeredY[i] - yPred[i] + xij * oldCoef;
                    rho += xij * residual;
                }
                rho /= nSamples;

                const newCoef = softThreshold(rho, this.alpha) / zj;
                const delta = newCoef - oldCoef;
                if (delta !== 0) {
                    for (let i = 0; i < nSamples; i++) {
                        yPred[i] += centeredX[i][j] * delta;
                    }
                }
                coef[j] = newCoef;
                const absDelta = Math.abs(delta);
                if (absDelta > maxUpdate) {
                    maxUpdate = absDelta;
                }
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
