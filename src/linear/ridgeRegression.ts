import { transpose, Inverse } from '../algebra';

function matMul(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const result: number[][] = [];
    for (let i = 0; i < rows; i++) {
        result.push([]);
        for (let j = 0; j < cols; j++) {
            let sum = 0;
            for (let k = 0; k < inner; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i].push(sum);
        }
    }
    return result;
}

export interface RidgeRegressionProps {
    alpha?: number;
    fitIntercept?: boolean;
}

export class RidgeRegression {
    private alpha: number;
    private fitIntercept: boolean;
    private coef: number[];
    private intercept: number;

    public constructor(props: RidgeRegressionProps = {}) {
        const { alpha = 1, fitIntercept = true } = props;
        if (!Number.isFinite(alpha) || alpha < 0) {
            throw new Error('alpha must be a finite number >= 0');
        }
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
        this.coef = [];
        this.intercept = 0;
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
        const Ymat = Y.map(v => [v]);
        const XT = transpose(Xb);
        const XTX = matMul(XT, Xb);

        for (let i = 0; i < XTX.length; i++) {
            if (this.fitIntercept && i === 0) {
                continue;
            }
            XTX[i][i] += this.alpha;
        }

        const XTY = matMul(XT, Ymat);
        const params = matMul(Inverse.elementary(XTX) as number[][], XTY);

        if (this.fitIntercept) {
            this.intercept = params[0][0];
            this.coef = params.slice(1).map(p => p[0]);
        } else {
            this.intercept = 0;
            this.coef = params.map(p => p[0]);
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
