import { lstsq } from '../algebra/lstsq';
import { RegressorBase } from '../base/regressor';
import { registerEstimator, Params } from '../base/estimator';

export interface PolynomialRegressionProps {
    degree?: number;
}

export class PolynomialRegression extends RegressorBase {
    private degree: number;
    private coef: number[];
    private intercept: number;
    private fitted: boolean;

    constructor(props: PolynomialRegressionProps = {}) {
        super();
        const { degree = 2 } = props;
        if (!Number.isInteger(degree) || degree < 1) {
            throw new Error('degree must be an integer >= 1');
        }
        this.degree = degree;
        this.coef = [];
        this.intercept = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return { degree: this.degree };
    }

    private transform(X: number[][]): number[][] {
        return X.map(row => {
            const features: number[] = [];
            for (const value of row) {
                for (let power = 1; power <= this.degree; power++) {
                    features.push(value ** power);
                }
            }
            return features;
        });
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
        const Xpoly = this.transform(X);
        const Xb = Xpoly.map(r => [1, ...r]);
        const params = lstsq(Xb, Y);
        if (params === false) {
            throw new Error(
                'polynomial design matrix is singular: too few samples for the requested degree or collinear features'
            );
        }

        this.intercept = params[0];
        this.coef = params.slice(1);
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('PolynomialRegression must be fitted before calling predict');
        }
        const Xpoly = this.transform(X);
        return Xpoly.map(row => {
            let sum = this.intercept;
            for (let i = 0; i < this.coef.length; i++) {
                sum += this.coef[i] * row[i];
            }
            return sum;
        });
    }
}
registerEstimator('PolynomialRegression', PolynomialRegression);
