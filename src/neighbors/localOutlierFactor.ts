import { OutlierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';

export interface LocalOutlierFactorProps {
    nNeighbors?: number;
    contamination?: number | 'auto';
    novelty?: boolean;
}

function distance(a: number[], b: number[]): number {
    let total = 0;
    for (let j = 0; j < a.length; j++) total += (a[j] - b[j]) ** 2;
    return Math.sqrt(total);
}

export class LocalOutlierFactor extends OutlierBase {
    private nNeighbors: number;
    private contamination: number | 'auto';
    private novelty: boolean;
    private trainX: number[][] = [];
    private kDistances: number[] = [];
    private localReachability: number[] = [];
    private negativeOutlierFactorState: number[] = [];
    private offsetState = -1.5;
    private effectiveNeighbors = 0;

    constructor(props: LocalOutlierFactorProps = {}) {
        super();
        const { nNeighbors = 20, contamination = 'auto', novelty = false } = props;
        if (!Number.isInteger(nNeighbors) || nNeighbors < 1) throw new Error('nNeighbors must be a positive integer');
        if (contamination !== 'auto' && (!(contamination > 0) || contamination > 0.5)) throw new Error('contamination must be auto or in (0, 0.5]');
        this.nNeighbors = nNeighbors;
        this.contamination = contamination;
        this.novelty = novelty;
    }

    public getParams(): Params { return { nNeighbors: this.nNeighbors, contamination: this.contamination, novelty: this.novelty }; }

    private neighbors(sample: number[], exclude = -1): Array<{ index: number; distance: number }> {
        return this.trainX.map((row, index) => ({ index, distance: index === exclude ? Infinity : distance(sample, row) }))
            .sort((a, b) => a.distance - b.distance || a.index - b.index)
            .slice(0, this.effectiveNeighbors);
    }

    public fit(X: number[][]): void {
        if (X.length < 2 || X[0].length === 0 || X.some(row => row.length !== X[0].length)) throw new Error('X must contain at least two rectangular samples');
        this.trainX = X.map(row => row.slice());
        this.effectiveNeighbors = Math.min(this.nNeighbors, X.length - 1);
        const neighborRows = X.map((row, i) => this.neighbors(row, i));
        this.kDistances = neighborRows.map(rows => rows[rows.length - 1].distance);
        this.localReachability = neighborRows.map(rows => {
            const meanReach = rows.reduce((sum, hit) => sum + Math.max(hit.distance, this.kDistances[hit.index]), 0) / rows.length;
            return 1 / (meanReach + 1e-10);
        });
        this.negativeOutlierFactorState = neighborRows.map((rows, i) => {
            const neighborDensity = rows.reduce((sum, hit) => sum + this.localReachability[hit.index], 0) / rows.length;
            const ratio = neighborDensity === Infinity && this.localReachability[i] === Infinity ? 1 : neighborDensity / this.localReachability[i];
            return -ratio;
        });
        if (this.contamination === 'auto') this.offsetState = -1.5;
        else {
            const ordered = this.negativeOutlierFactorState.slice().sort((a, b) => a - b);
            const pos = (ordered.length - 1) * this.contamination;
            const lo = Math.floor(pos), hi = Math.ceil(pos);
            this.offsetState = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo);
        }
    }

    public scoreSamples(X: number[][]): number[] {
        if (this.trainX.length === 0) throw new Error('LocalOutlierFactor is not fitted');
        if (!this.novelty) throw new Error('scoreSamples is available only when novelty=true; use negativeOutlierFactor for training data');
        return X.map(sample => {
            const rows = this.neighbors(sample);
            const meanReach = rows.reduce((sum, hit) => sum + Math.max(hit.distance, this.kDistances[hit.index]), 0) / rows.length;
            const lrd = 1 / (meanReach + 1e-10);
            const neighborDensity = rows.reduce((sum, hit) => sum + this.localReachability[hit.index], 0) / rows.length;
            return -(neighborDensity === Infinity && lrd === Infinity ? 1 : neighborDensity / lrd);
        });
    }
    public decisionFunction(X: number[][]): number[] { return this.scoreSamples(X).map(score => score - this.offsetState); }
    public predict(X: number[][]): number[] {
        if (!this.novelty) throw new Error('predict is available only when novelty=true; use fitPredict for training data');
        return this.decisionFunction(X).map(score => score < 0 ? -1 : 1);
    }
    public fitPredict(X: number[][]): number[] {
        if (this.novelty) throw new Error('fitPredict is unavailable when novelty=true; call fit then predict on new data');
        this.fit(X);
        return this.negativeOutlierFactorState.map(score => score < this.offsetState ? -1 : 1);
    }
    public get negativeOutlierFactor(): number[] { return this.negativeOutlierFactorState.slice(); }
    public get offset(): number { return this.offsetState; }
}
registerEstimator('LocalOutlierFactor', LocalOutlierFactor);
