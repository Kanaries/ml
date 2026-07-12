import { lstsq } from '../algebra/lstsq';
import { RegressorBase } from '../base/regressor';
import { registerEstimator, Params } from '../base/estimator';

export interface RidgeRegressionProps {
    alpha?: number;
    fitIntercept?: boolean;
}

export class RidgeRegression extends RegressorBase {
    private alpha: number;
    private fitIntercept: boolean;
    private coef: number[];
    private intercept: number;

    public constructor(props: RidgeRegressionProps = {}) {
        super();
        const { alpha = 1, fitIntercept = true } = props;
        if (!Number.isFinite(alpha) || alpha < 0) {
            throw new Error('alpha must be a finite number >= 0');
        }
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
        this.coef = [];
        this.intercept = 0;
    }

    public getParams(): Params {
        return { alpha: this.alpha, fitIntercept: this.fitIntercept };
    }

    public fit(X: number[][], Y: number[]): void {
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

        const Xb = this.fitIntercept ? X.map(r => [1, ...r]) : X.map(r => [...r]);
        const nCols = Xb[0].length;

        // ridge as augmented least squares: append sqrt(alpha) * e_j rows for
        // every penalized coefficient (the intercept is not penalized), then
        // solve min ||[X; sqrt(a) D] w - [y; 0]|| by QR — equivalent to
        // (X'X + alpha D) w = X'y without squaring the condition number
        const rows = Xb.map(r => r.slice());
        const target = Y.slice();
        if (this.alpha > 0) {
            const sqrtAlpha = Math.sqrt(this.alpha);
            for (let j = this.fitIntercept ? 1 : 0; j < nCols; j++) {
                const row = new Array(nCols).fill(0);
                row[j] = sqrtAlpha;
                rows.push(row);
                target.push(0);
            }
        }
        const params = lstsq(rows, target);
        if (params === false) {
            throw new Error(
                'X is singular: features are collinear; use alpha > 0 to regularize'
            );
        }

        if (this.fitIntercept) {
            this.intercept = params[0];
            this.coef = params.slice(1);
        } else {
            this.intercept = 0;
            this.coef = params.slice();
        }
    }

    public predict(X: number[][]): number[] {
        if (this.coef.length === 0) {
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
registerEstimator('RidgeRegression', RidgeRegression);
