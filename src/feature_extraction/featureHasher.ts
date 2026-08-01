import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix } from '../data';
import { hashedRows } from './hashing';

export type HasherInput = 'dict' | 'pair' | 'string';
export interface FeatureHasherProps { nFeatures?: number; inputType?: HasherInput; alternateSign?: boolean; }

export class FeatureHasher extends TransformerBase<any, any> {
    private nFeatures: number; private inputType: HasherInput; private alternateSign: boolean;
    constructor(props: FeatureHasherProps = {}) { super(); const { nFeatures = 2 ** 20, inputType = 'dict', alternateSign = true } = props; if (!Number.isInteger(nFeatures) || nFeatures < 1 || !['dict', 'pair', 'string'].includes(inputType)) throw new Error('invalid FeatureHasher parameters'); this.nFeatures = nFeatures; this.inputType = inputType; this.alternateSign = alternateSign; }
    public getParams(): Params { return { nFeatures: this.nFeatures, inputType: this.inputType, alternateSign: this.alternateSign }; }
    public fit(_X: any[]): void {}
    public transform(X: any[]): CSRMatrix {
        const rows = X.map(sample => {
            if (this.inputType === 'dict') {
                if (sample === null || typeof sample !== 'object' || Array.isArray(sample)) throw new Error('dict input expects plain objects');
                return Object.entries(sample).map<[string, number]>(([key, value]) => {
                    if (typeof value === 'string') return [`${key}=${value}`, 1];
                    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('dict feature values must be finite numbers or strings');
                    return [key, value];
                });
            }
            if (!Array.isArray(sample)) throw new Error(`${this.inputType} input expects arrays`);
            if (this.inputType === 'string') return sample.map(value => { if (typeof value !== 'string') throw new Error('string features must be strings'); return [value, 1] as [string, number]; });
            return sample.map(value => { if (!Array.isArray(value) || value.length !== 2 || typeof value[0] !== 'string' || typeof value[1] !== 'number' || !Number.isFinite(value[1])) throw new Error('pair features must be [string, number]'); return [value[0], value[1]] as [string, number]; });
        });
        return hashedRows(rows, this.nFeatures, this.alternateSign);
    }
}
registerEstimator('FeatureHasher', FeatureHasher);
