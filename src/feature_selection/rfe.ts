import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { FeatureSelectingEstimator, estimatorImportancesSquared, resolveFeatureCount, resolveStep, selectColumns, validateSelectionInput } from './common';

export interface RFEProps {
    estimator: FeatureSelectingEstimator;
    nFeaturesToSelect?: number;
    step?: number;
}

export class RFE extends BaseEstimator {
    protected estimator: FeatureSelectingEstimator;
    protected nFeaturesToSelect?: number;
    protected step: number;
    protected supportIndices: number[] = [];
    protected rankingState: number[] = [];
    protected fittedEstimator?: FeatureSelectingEstimator;
    protected nFeatures = 0;

    constructor(props: RFEProps) {
        super();
        if (!props?.estimator) throw new Error('RFE requires an estimator');
        this.estimator = props.estimator;
        this.nFeaturesToSelect = props.nFeaturesToSelect;
        this.step = props.step ?? 1;
    }

    public getParams(): Params {
        return { estimator: this.estimator, nFeaturesToSelect: this.nFeaturesToSelect, step: this.step };
    }

    public fit(X: number[][], y?: number[]): void {
        if (!y) throw new Error('RFE requires y');
        this.nFeatures = validateSelectionInput(X, y);
        const target = resolveFeatureCount(this.nFeaturesToSelect, this.nFeatures, Math.floor(this.nFeatures / 2) || 1);
        const active = Array.from({ length: this.nFeatures }, (_, i) => i);
        const fixedStep = resolveStep(this.step, this.nFeatures);
        this.rankingState = new Array(this.nFeatures).fill(1);
        while (active.length > target) {
            const fitted = this.estimator.clone() as FeatureSelectingEstimator;
            fitted.fit(selectColumns(X, active), y);
            const importance = estimatorImportancesSquared(fitted);
            if (importance.length !== active.length) throw new Error('estimator importance length does not match the active feature count');
            const removeCount = Math.min(fixedStep, active.length - target);
            const remove = importance.map((value, i) => ({ value, i }))
                .sort((a, b) => a.value - b.value || a.i - b.i)
                .slice(0, removeCount)
                .map(entry => entry.i)
                .sort((a, b) => b - a);
            for (let i = 0; i < this.rankingState.length; i++) {
                if (this.rankingState[i] > 1) this.rankingState[i]++;
            }
            for (const local of remove) {
                this.rankingState[active[local]] = 2;
                active.splice(local, 1);
            }
        }
        this.supportIndices = active;
        this.fittedEstimator = this.estimator.clone() as FeatureSelectingEstimator;
        this.fittedEstimator.fit(selectColumns(X, active), y);
    }

    public transform(X: number[][]): number[][] {
        if (!this.fittedEstimator) throw new Error('RFE is not fitted');
        if (X.some(row => row.length !== this.nFeatures)) throw new Error('input feature size does not match fitted selector');
        return selectColumns(X, this.supportIndices);
    }

    public predict(X: number[][]): number[] {
        if (!this.fittedEstimator?.predict) throw new Error('fitted estimator does not implement predict');
        return this.fittedEstimator.predict(this.transform(X));
    }

    public score(X: number[][], y: number[]): number {
        if (!this.fittedEstimator?.score) throw new Error('fitted estimator does not implement score');
        return this.fittedEstimator.score(this.transform(X), y);
    }

    public getSupport(indices = false): boolean[] | number[] {
        if (indices) return this.supportIndices.slice();
        const selected = new Set(this.supportIndices);
        return Array.from({ length: this.nFeatures }, (_, i) => selected.has(i));
    }

    public get ranking(): number[] { return this.rankingState.slice(); }
}
registerEstimator('RFE', RFE);
