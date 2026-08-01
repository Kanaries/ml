import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, NumericMatrix, matrixRow, matrixShape } from '../data';

export type FeatureDictionary = Record<string, number | string>;
export interface DictVectorizerProps { sparse?: boolean; sort?: boolean; separator?: string; }

export class DictVectorizer extends TransformerBase<any, any> {
    private sparse: boolean; private sort: boolean; private separator: string; private vocabularyState = new Map<string, number>();
    constructor(props: DictVectorizerProps = {}) { super(); const { sparse = true, sort = true, separator = '=' } = props; this.sparse = sparse; this.sort = sort; this.separator = separator; }
    public getParams(): Params { return { sparse: this.sparse, sort: this.sort, separator: this.separator }; }
    private entries(sample: FeatureDictionary): Array<[string, number]> {
        if (sample === null || typeof sample !== 'object' || Array.isArray(sample)) throw new Error('DictVectorizer expects plain feature dictionaries');
        return Object.entries(sample).map(([key, value]) => typeof value === 'string' ? [`${key}${this.separator}${value}`, 1] : [key, value]).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)) as Array<[string, number]>;
    }
    public fit(X: FeatureDictionary[]): void {
        const features = new Set<string>(); for (const sample of X) for (const [feature] of this.entries(sample)) features.add(feature);
        let names = Array.from(features); if (this.sort) names.sort(); this.vocabularyState = new Map(names.map((name, index) => [name, index]));
    }
    public transform(X: FeatureDictionary[]): CSRMatrix | number[][] {
        if (this.vocabularyState.size === 0) throw new Error('DictVectorizer is not fitted');
        const dense = X.map(sample => { const row = new Array(this.vocabularyState.size).fill(0); for (const [feature, value] of this.entries(sample)) { const index = this.vocabularyState.get(feature); if (index !== undefined) row[index] += value; } return row; });
        return this.sparse ? CSRMatrix.fromDense(dense) : dense;
    }
    public inverseTransform(X: NumericMatrix): Array<Record<string, number>> {
        if (this.vocabularyState.size === 0) throw new Error('DictVectorizer is not fitted');
        const [rows, columns] = matrixShape(X); if (columns !== this.vocabularyState.size) throw new Error('input feature count differs from fitted vocabulary');
        const names = this.getFeatureNamesOut();
        return Array.from({ length: rows }, (_, i) => { const result: Record<string, number> = {}; matrixRow(X, i).forEach((value, j) => { if (value !== 0) result[names[j]] = value; }); return result; });
    }
    public restrict(features: number[] | boolean[], support = true): this {
        if (this.vocabularyState.size === 0) throw new Error('DictVectorizer is not fitted');
        const width = this.vocabularyState.size;
        let selected: number[];
        if (support) {
            if (features.length !== width || features.some(value => typeof value !== 'boolean')) throw new Error('support mask must contain one boolean per feature');
            selected = features.flatMap((value, index) => value ? [index] : []);
        } else {
            if (features.some(value => typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value >= width)) throw new Error('feature indices are out of bounds');
            selected = (features as number[]).map(Number);
        }
        const selectedSet = new Set(selected), names = this.getFeatureNamesOut();
        this.vocabularyState = new Map(names.filter((_, index) => selectedSet.has(index)).map((name, index) => [name, index]));
        return this;
    }
    public getFeatureNamesOut(): string[] { return Array.from(this.vocabularyState).sort((a, b) => a[1] - b[1]).map(([name]) => name); }
    public get vocabulary(): ReadonlyMap<string, number> { return new Map(this.vocabularyState); }
}
registerEstimator('DictVectorizer', DictVectorizer);
