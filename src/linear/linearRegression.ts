import { lstsq } from '../algebra/lstsq';

export class LinearRegression {
    private coef: number[];
    private intercept: number;
    private fitted: boolean;
    public constructor() {
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
        const nFeatures = X[0].length;
        for (const row of X) {
            if (row.length !== nFeatures) {
                throw new Error('all rows in X must have the same length');
            }
        }
        const Xb = X.map(r => [1, ...r]);
        // QR least squares on [1 X] directly — no normal equations, no
        // explicit inverse, so near-collinear designs stay accurate
        const params = lstsq(Xb, Y);
        if (params === false) {
            throw new Error(
                'X is singular: features are collinear or n_samples < n_features + 1; consider Ridge (alpha > 0)'
            );
        }
        this.intercept = params[0];
        this.coef = params.slice(1);
        this.fitted = true;
    }
    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('LinearRegression must be fitted before calling predict');
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
