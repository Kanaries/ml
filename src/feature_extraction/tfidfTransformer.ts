import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, forEachNonZeroInRow, isCSRMatrix, matrixShape, NumericMatrix } from '../data';

export interface TfidfTransformerProps {
    norm?: 'l1' | 'l2' | null;
    useIdf?: boolean;
    smoothIdf?: boolean;
    sublinearTf?: boolean;
}

export class TfidfTransformer extends TransformerBase<NumericMatrix, CSRMatrix> {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
    private norm: 'l1' | 'l2' | null;
    private useIdf: boolean;
    private smoothIdf: boolean;
    private sublinearTf: boolean;
    private idfState: number[] = [];

    constructor(props: TfidfTransformerProps = {}) {
        super();
        const { norm = 'l2', useIdf = true, smoothIdf = true, sublinearTf = false } = props;
        if (norm !== null && norm !== 'l1' && norm !== 'l2') throw new Error('norm must be l1, l2, or null');
        if (typeof useIdf !== 'boolean' || typeof smoothIdf !== 'boolean' || typeof sublinearTf !== 'boolean') throw new Error('TF-IDF flags must be booleans');
        this.norm = norm; this.useIdf = useIdf; this.smoothIdf = smoothIdf; this.sublinearTf = sublinearTf;
    }
    public getParams(): Params { return { norm: this.norm, useIdf: this.useIdf, smoothIdf: this.smoothIdf, sublinearTf: this.sublinearTf }; }

    private validateCounts(X: NumericMatrix): [number, number] {
        const shape = matrixShape(X);
        if (!isCSRMatrix(X) && X.some(row => row.length !== shape[1])) throw new Error('X must be a rectangular count matrix');
        for (let row = 0; row < shape[0]; row++) forEachNonZeroInRow(X, row, (_column, value) => {
            if (!Number.isFinite(value) || value < 0) throw new Error('TfidfTransformer requires finite non-negative term counts');
        });
        return shape;
    }

    public fit(X: NumericMatrix): void {
        const [nSamples, nFeatures] = this.validateCounts(X);
        if (nSamples === 0 || nFeatures === 0) throw new Error('X must be non-empty');
        const df = new Array(nFeatures).fill(0);
        for (let row = 0; row < nSamples; row++) forEachNonZeroInRow(X, row, column => { df[column]++; });
        const smooth = this.smoothIdf ? 1 : 0;
        this.idfState = df.map(value => this.useIdf ? Math.log((nSamples + smooth) / (value + smooth)) + 1 : 1);
    }

    public transform(X: NumericMatrix): CSRMatrix {
        const [nSamples, nFeatures] = this.validateCounts(X);
        if (this.idfState.length === 0) throw new Error('TfidfTransformer is not fitted');
        if (nFeatures !== this.idfState.length) throw new Error('input feature count does not match fitted transformer');
        const data: number[] = [], indices: number[] = [], indptr = [0];
        for (let row = 0; row < nSamples; row++) {
            const rowValues: Array<[number, number]> = [];
            forEachNonZeroInRow(X, row, (column, count) => {
                const tf = this.sublinearTf ? 1 + Math.log(count) : count;
                rowValues.push([column, tf * this.idfState[column]]);
            });
            let scale = 1;
            if (this.norm === 'l1') scale = rowValues.reduce((sum, [, value]) => sum + Math.abs(value), 0) || 1;
            if (this.norm === 'l2') scale = Math.sqrt(rowValues.reduce((sum, [, value]) => sum + value * value, 0)) || 1;
            for (const [column, value] of rowValues) {
                const normalized = value / scale;
                if (normalized !== 0) { indices.push(column); data.push(normalized); }
            }
            indptr.push(data.length);
        }
        return new CSRMatrix(data, indices, indptr, [nSamples, nFeatures]);
    }
    public get idf(): number[] { if (!this.useIdf) throw new Error('idf is unavailable when useIdf=false'); if (this.idfState.length === 0) throw new Error('TfidfTransformer is not fitted'); return this.idfState.slice(); }
}
registerEstimator('TfidfTransformer', TfidfTransformer);
