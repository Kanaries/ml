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

export interface PolynomialRegressionProps {
    degree?: number;
}

export class PolynomialRegression {
    private degree: number;
    private coef: number[];
    private intercept: number;

    constructor(props: PolynomialRegressionProps = {}) {
        const { degree = 2 } = props;
        if (!Number.isInteger(degree) || degree < 1) {
            throw new Error('degree must be an integer >= 1');
        }
        this.degree = degree;
        this.coef = [];
        this.intercept = 0;
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

        const Xpoly = this.transform(X);
        const Xb = Xpoly.map(r => [1, ...r]);
        const Ymat = Y.map(v => [v]);
        const XT = transpose(Xb);
        const XTX = matMul(XT, Xb);
        const XTXInv = Inverse.elementary(XTX) as number[][];
        const XTY = matMul(XT, Ymat);
        const params = matMul(XTXInv, XTY);

        this.intercept = params[0][0];
        this.coef = params.slice(1).map(p => p[0]);
    }

    public predict(X: number[][]): number[] {
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
