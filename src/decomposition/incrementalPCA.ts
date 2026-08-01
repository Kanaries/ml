import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { jacobiSVD } from '../discriminant_analysis/linalg';

export interface IncrementalPCAProps { nComponents?: number | null; batchSize?: number; }

export class IncrementalPCA extends TransformerBase {
    private nComponents: number | null;
    private batchSize?: number;
    private nSamplesSeenState = 0;
    private meanState: number[] = [];
    private varianceState: number[] = [];
    private componentsState: number[][] = [];
    private explainedVarianceState: number[] = [];
    private singularValuesState: number[] = [];

    constructor(props: IncrementalPCAProps = {}) {
        super();
        const { nComponents = null, batchSize } = props;
        if (nComponents !== null && (!Number.isInteger(nComponents) || nComponents < 1)) throw new Error('nComponents must be null or positive');
        if (batchSize !== undefined && (!Number.isInteger(batchSize) || batchSize < 1)) throw new Error('batchSize must be positive');
        this.nComponents = nComponents; this.batchSize = batchSize;
    }
    public getParams(): Params { return { nComponents: this.nComponents, batchSize: this.batchSize }; }
    private reset(): void { this.nSamplesSeenState = 0; this.meanState = []; this.varianceState = []; this.componentsState = []; this.explainedVarianceState = []; this.singularValuesState = []; }
    public partialFit(X: number[][]): this {
        if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(value => !Number.isFinite(value)))) throw new Error('X must be a non-empty finite rectangular matrix');
        const n = X.length, p = X[0].length;
        const first = this.nSamplesSeenState === 0;
        if (!first && this.meanState.length !== p) throw new Error('feature count differs from previous partialFit calls');
        const k = first ? (this.nComponents ?? Math.min(n, p)) : this.componentsState.length;
        if (first && k > Math.min(n, p)) throw new Error('nComponents must not exceed the first batch sample or feature count');
        const batchMean = Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n);
        const batchVariance = Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + (row[j] - batchMean[j]) ** 2, 0) / n);
        const oldCount = this.nSamplesSeenState, total = oldCount + n;
        const oldMean = first ? new Array(p).fill(0) : this.meanState.slice();
        const combinedMean = Array.from({ length: p }, (_, j) => (oldCount * oldMean[j] + n * batchMean[j]) / total);
        const combinedVariance = Array.from({ length: p }, (_, j) => {
            const oldM2 = first ? 0 : this.varianceState[j] * oldCount;
            const batchM2 = batchVariance[j] * n;
            const delta = batchMean[j] - oldMean[j];
            return (oldM2 + batchM2 + delta * delta * oldCount * n / total) / total;
        });
        let augmented = X.map(row => row.map((value, j) => value - batchMean[j]));
        if (!first) {
            const retained = this.componentsState.map((component, c) => component.map(value => value * this.singularValuesState[c]));
            const scale = Math.sqrt(oldCount * n / total);
            const meanCorrection = oldMean.map((value, j) => scale * (value - batchMean[j]));
            augmented = [...retained, ...augmented, meanCorrection];
        }
        const { S, Vt } = jacobiSVD(augmented);
        this.singularValuesState = S.slice(0, k);
        this.componentsState = Vt.slice(0, k).map(vector => {
            const out = vector.slice(); let pivot = 0;
            for (let j = 1; j < out.length; j++) if (Math.abs(out[j]) > Math.abs(out[pivot])) pivot = j;
            if (out[pivot] < 0) for (let j = 0; j < out.length; j++) out[j] *= -1;
            return out;
        });
        this.explainedVarianceState = this.singularValuesState.map(value => value * value / Math.max(1, total - 1));
        this.meanState = combinedMean;
        this.varianceState = combinedVariance;
        this.nSamplesSeenState = total;
        return this;
    }
    public fit(X: number[][]): void {
        this.reset();
        if (X.length === 0 || X[0].length === 0) throw new Error('X must be a non-empty rectangular matrix');
        const batch = this.batchSize ?? Math.max(1, 5 * X[0].length);
        const minimum = this.nComponents ?? 0;
        for (let start = 0; start < X.length;) {
            let end = Math.min(start + batch, X.length);
            if (X.length - end > 0 && X.length - end < minimum) end = X.length;
            this.partialFit(X.slice(start, end));
            start = end;
        }
    }
    public transform(X: number[][]): number[][] {
        if (this.componentsState.length === 0) throw new Error('IncrementalPCA is not fitted');
        if (X.some(row => row.length !== this.meanState.length || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted feature count');
        return X.map(row => this.componentsState.map(component => component.reduce((sum, value, j) => sum + value * (row[j] - this.meanState[j]), 0)));
    }
    public inverseTransform(X: number[][]): number[][] {
        if (this.componentsState.length === 0) throw new Error('IncrementalPCA is not fitted');
        if (X.some(row => row.length !== this.componentsState.length || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted component count');
        return X.map(row => this.meanState.map((mean, j) => mean + row.reduce((sum, value, c) => sum + value * this.componentsState[c][j], 0)));
    }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
    public get mean(): number[] { return this.meanState.slice(); }
    public get explainedVariance(): number[] { return this.explainedVarianceState.slice(); }
    public get singularValues(): number[] { return this.singularValuesState.slice(); }
    public get nSamplesSeen(): number { return this.nSamplesSeenState; }
}
registerEstimator('IncrementalPCA', IncrementalPCA);
