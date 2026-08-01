import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { FeatureSelectingEstimator, estimatorImportances, selectColumns, validateSelectionInput } from './common';

export interface SelectFromModelProps {
    estimator: FeatureSelectingEstimator;
    threshold?: number | 'mean' | 'median' | `${number}*mean` | `${number}*median`;
    maxFeatures?: number;
}

export class SelectFromModel extends TransformerBase<number[][], number[][]> {
    private estimator: FeatureSelectingEstimator;
    private threshold: SelectFromModelProps['threshold'] | number;
    private maxFeatures?: number;
    private supportIndices: number[] = [];
    private nFeatures = 0;
    private fittedEstimatorState?: FeatureSelectingEstimator;

    constructor(props: SelectFromModelProps) {
        super();
        if (!props?.estimator) throw new Error('SelectFromModel requires an estimator');
        this.estimator = props.estimator;
        // sklearn treats threshold=None + max_features as pure top-k.
        this.threshold = props.threshold ?? (props.maxFeatures === undefined ? 'mean' : -Infinity);
        this.maxFeatures = props.maxFeatures;
        if (this.maxFeatures !== undefined && (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)) {
            throw new Error('maxFeatures must be a positive integer');
        }
    }

    public getParams(): Params {
        return { estimator: this.estimator, threshold: this.threshold, maxFeatures: this.maxFeatures };
    }

    public fit(X: number[][], y?: number[]): void {
        if (!y) throw new Error('SelectFromModel requires y');
        this.nFeatures = validateSelectionInput(X, y);
        const fitted = this.estimator.clone() as FeatureSelectingEstimator;
        fitted.fit(X, y);
        this.fittedEstimatorState = fitted;
        const importance = estimatorImportances(fitted);
        if (importance.length !== this.nFeatures) throw new Error('estimator importance length does not match X');
        const ordered = importance.slice().sort((a, b) => a - b);
        const mean = importance.reduce((sum, value) => sum + value, 0) / importance.length;
        const median = (ordered[Math.floor((ordered.length - 1) / 2)] + ordered[Math.ceil((ordered.length - 1) / 2)]) / 2;
        let threshold: number;
        if (typeof this.threshold === 'number') threshold = this.threshold;
        else {
            const match = /^(?:(\d+(?:\.\d+)?)\*)?(mean|median)$/.exec(this.threshold!);
            if (!match) throw new Error('threshold must be numeric, mean, median, k*mean, or k*median');
            threshold = (match[1] === undefined ? 1 : Number(match[1])) * (match[2] === 'mean' ? mean : median);
        }
        this.supportIndices = importance.map((value, index) => ({ value, index }))
            .filter(entry => entry.value >= threshold)
            .sort((a, b) => b.value - a.value || a.index - b.index)
            .slice(0, this.maxFeatures)
            .map(entry => entry.index)
            .sort((a, b) => a - b);
    }

    public transform(X: number[][]): number[][] {
        if (this.nFeatures === 0) throw new Error('SelectFromModel is not fitted');
        if (X.some(row => row.length !== this.nFeatures)) throw new Error('input feature size does not match fitted selector');
        return selectColumns(X, this.supportIndices);
    }

    public getSupport(indices = false): boolean[] | number[] {
        if (indices) return this.supportIndices.slice();
        const selected = new Set(this.supportIndices);
        return Array.from({ length: this.nFeatures }, (_, i) => selected.has(i));
    }
    public get fittedEstimator(): FeatureSelectingEstimator {
        if (!this.fittedEstimatorState) throw new Error('SelectFromModel is not fitted');
        return this.fittedEstimatorState;
    }
}
registerEstimator('SelectFromModel', SelectFromModel);
