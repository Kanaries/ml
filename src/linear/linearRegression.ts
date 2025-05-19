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

export class LinearRegression {
    private coef: number[];
    private intercept: number;
    public constructor() {
        this.coef = [];
        this.intercept = 0;
    }
    public fit(X: number[][], Y: number[]): void {
        const Xb = X.map(r => [1, ...r]);
        const Ymat = Y.map(v => [v]);
        const XT = transpose(Xb);
        const XTX = matMul(XT, Xb);
        const XTX_inv = Inverse.elementary(XTX) as number[][];
        const XTY = matMul(XT, Ymat);
        const params = matMul(XTX_inv, XTY);
        this.intercept = params[0][0];
        this.coef = params.slice(1).map(p => p[0]);
    }
    public predict(X: number[][]): number[] {
        return X.map(row => {
            let sum = this.intercept;
            for (let i = 0; i < this.coef.length; i++) {
                sum += this.coef[i] * row[i];
            }
            return sum;
        });
    }
}
