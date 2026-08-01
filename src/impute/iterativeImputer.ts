import { BaseEstimator, TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { RidgeRegression } from '../linear';
import { splitEstimatorParams } from '../multioutput/common';
import { createRandomGenerator } from '../utils/random';

interface RegressorLike extends BaseEstimator { fit(X: number[][], y: number[]): void; predict(X: number[][]): number[]; }
interface ImputationStep { feature: number; estimator: RegressorLike; }
export interface IterativeImputerProps {
    estimator?: RegressorLike;
    maxIter?: number;
    tol?: number;
    initialStrategy?: 'mean' | 'median' | 'mostFrequent' | 'constant';
    fillValue?: number;
    imputationOrder?: 'ascending' | 'descending' | 'roman' | 'arabic' | 'random';
    skipComplete?: boolean;
    minValue?: number;
    maxValue?: number;
    randomState?: number;
}

function statistic(values: number[], strategy: IterativeImputerProps['initialStrategy'], fillValue: number): number {
    if (values.length === 0 || strategy === 'constant') return fillValue;
    if (strategy === 'mean') return values.reduce((a, b) => a + b, 0) / values.length;
    const sorted = values.slice().sort((a, b) => a - b);
    if (strategy === 'median') return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2;
    const counts = new Map<number, number>(); for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
    return Array.from(counts).sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
}

export class IterativeImputer extends TransformerBase {
    private estimator: RegressorLike;
    private maxIter: number;
    private tol: number;
    private initialStrategy: NonNullable<IterativeImputerProps['initialStrategy']>;
    private fillValue: number;
    private imputationOrder: NonNullable<IterativeImputerProps['imputationOrder']>;
    private skipComplete: boolean;
    private minValue: number;
    private maxValue: number;
    private randomState?: number;
    private initialStatistics: number[] = [];
    private sequenceState: ImputationStep[] = [];
    private nIterState = 0;
    private nFeaturesState = 0;
    constructor(props: IterativeImputerProps = {}) {
        super();
        const { estimator = new RidgeRegression({ alpha: 1e-6 }), maxIter = 10, tol = 1e-3, initialStrategy = 'mean', fillValue = 0, imputationOrder = 'ascending', skipComplete = false, minValue = -Infinity, maxValue = Infinity, randomState } = props;
        if (!(estimator instanceof BaseEstimator) || typeof estimator.predict !== 'function') throw new Error('IterativeImputer estimator must implement fit and predict');
        if (!['mean', 'median', 'mostFrequent', 'constant'].includes(initialStrategy) || !['ascending', 'descending', 'roman', 'arabic', 'random'].includes(imputationOrder) || !Number.isInteger(maxIter) || maxIter < 0 || !Number.isFinite(tol) || tol < 0 || !Number.isFinite(fillValue) || Number.isNaN(minValue) || Number.isNaN(maxValue) || minValue > maxValue) throw new Error('invalid IterativeImputer parameters');
        this.estimator = estimator; this.maxIter = maxIter; this.tol = tol; this.initialStrategy = initialStrategy;
        this.fillValue = fillValue; this.imputationOrder = imputationOrder; this.skipComplete = skipComplete;
        this.minValue = minValue; this.maxValue = maxValue; this.randomState = randomState;
    }
    public getParams(): Params { return { estimator: this.estimator, maxIter: this.maxIter, tol: this.tol, initialStrategy: this.initialStrategy, fillValue: this.fillValue, imputationOrder: this.imputationOrder, skipComplete: this.skipComplete, minValue: this.minValue, maxValue: this.maxValue, randomState: this.randomState }; }
    public setParams(params: Params): this { const { own, nested } = splitEstimatorParams(params, this.constructor.name); const next = { ...this.getParams(), ...own }; const estimator = (next.estimator as BaseEstimator).clone(); if (Object.keys(nested).length > 0) estimator.setParams(nested); next.estimator = estimator; return super.setParams(next); }
    private validate(X: number[][]): void { if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(value => typeof value !== 'number' || (!Number.isFinite(value) && !Number.isNaN(value))))) throw new Error('X must be a non-empty rectangular numeric matrix containing only finite values or NaN'); }
    private initialize(X: number[][]): { filled: number[][]; missing: boolean[][] } {
        const missing = X.map(row => row.map(Number.isNaN));
        const filled = X.map(row => row.slice());
        for (let feature = 0; feature < this.nFeaturesState; feature++) for (let i = 0; i < X.length; i++) if (missing[i][feature]) filled[i][feature] = this.initialStatistics[feature];
        return { filled, missing };
    }
    private featureOrder(missing: boolean[][], random?: () => number): number[] {
        const counts = Array.from({ length: this.nFeaturesState }, (_, feature) => missing.reduce((sum, row) => sum + (row[feature] ? 1 : 0), 0));
        let order = Array.from({ length: this.nFeaturesState }, (_, i) => i).filter(feature => !this.skipComplete || counts[feature] > 0);
        if (this.imputationOrder === 'ascending') order.sort((a, b) => counts[a] - counts[b] || a - b);
        else if (this.imputationOrder === 'descending') order.sort((a, b) => counts[b] - counts[a] || b - a);
        else if (this.imputationOrder === 'arabic') order.reverse();
        else if (this.imputationOrder === 'random') { const draw = random ?? createRandomGenerator(this.randomState); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(draw() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } }
        return order;
    }
    private predictors(row: number[], feature: number): number[] { return row.filter((_, j) => j !== feature); }
    public fit(X: number[][]): void { this.fitTransform(X); }
    public fitTransform(X: number[][]): number[][] {
        this.validate(X); this.nFeaturesState = X[0].length;
        this.initialStatistics = Array.from({ length: this.nFeaturesState }, (_, feature) => statistic(X.map(row => row[feature]).filter(Number.isFinite), this.initialStrategy, this.fillValue));
        let { filled, missing } = this.initialize(X); this.sequenceState = []; this.nIterState = 0;
        if (this.nFeaturesState === 1 || this.maxIter === 0) return filled.map(row => row.slice());
        let observedScale = 0;
        for (const row of X) for (const value of row) if (Number.isFinite(value)) observedScale = Math.max(observedScale, Math.abs(value));
        const convergenceScale = observedScale;
        const random = this.imputationOrder === 'random' ? createRandomGenerator(this.randomState) : undefined;
        for (let iteration = 1; iteration <= this.maxIter; iteration++) {
            const previous = filled.map(row => row.slice()); this.nIterState = iteration;
            const order = this.featureOrder(missing, random);
            for (const feature of order) {
                const observedRows = X.map((row, i) => ({ row, i })).filter(entry => !missing[entry.i][feature]);
                if (observedRows.length === 0) continue;
                const member = this.estimator.clone() as RegressorLike;
                member.fit(observedRows.map(entry => this.predictors(filled[entry.i], feature)), observedRows.map(entry => entry.row[feature]));
                const missingRows = X.map((_, i) => i).filter(i => missing[i][feature]);
                if (missingRows.length > 0) {
                    const predictions = member.predict(missingRows.map(i => this.predictors(filled[i], feature)));
                    missingRows.forEach((row, i) => { filled[row][feature] = Math.max(this.minValue, Math.min(this.maxValue, predictions[i])); });
                }
                this.sequenceState.push({ feature, estimator: member });
            }
            let change = 0;
            for (let i = 0; i < X.length; i++) {
                let rowChange = 0;
                for (let j = 0; j < this.nFeaturesState; j++) if (missing[i][j]) rowChange += Math.abs(filled[i][j] - previous[i][j]);
                change = Math.max(change, rowChange);
            }
            if (change < this.tol * convergenceScale) break;
        }
        return filled;
    }
    public transform(X: number[][]): number[][] {
        this.validate(X); if (this.initialStatistics.length === 0) throw new Error('IterativeImputer is not fitted');
        if (X[0].length !== this.nFeaturesState) throw new Error('input feature count differs from fitted imputer');
        const { filled, missing } = this.initialize(X);
        for (const step of this.sequenceState) {
            const rows = X.map((_, i) => i).filter(i => missing[i][step.feature]);
            if (rows.length === 0) continue;
            const predictions = step.estimator.predict(rows.map(i => this.predictors(filled[i], step.feature)));
            rows.forEach((row, i) => { filled[row][step.feature] = Math.max(this.minValue, Math.min(this.maxValue, predictions[i])); });
        }
        return filled;
    }
    public get imputationSequence(): ImputationStep[] { return this.sequenceState.slice(); }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('IterativeImputer', IterativeImputer);
