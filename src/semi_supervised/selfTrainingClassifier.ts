import { BaseEstimator, ClassifierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { splitEstimatorParams } from '../multioutput/common';

interface ProbabilisticClassifier extends BaseEstimator {
    fit(X: number[][], y: number[]): void;
    predict(X: number[][]): number[];
    predictProba(X: number[][]): number[][];
}
export interface SelfTrainingClassifierProps {
    estimator: ProbabilisticClassifier;
    threshold?: number;
    criterion?: 'threshold' | 'kBest';
    kBest?: number;
    maxIter?: number;
}

export class SelfTrainingClassifier extends ClassifierBase {
    private estimator: ProbabilisticClassifier;
    private threshold: number;
    private criterion: 'threshold' | 'kBest';
    private kBest: number;
    private maxIter: number;
    private fittedEstimator?: ProbabilisticClassifier;
    private transductionState: number[] = [];
    private labeledIterationState: number[] = [];
    private nIterState = 0;
    private terminationConditionState: 'maxIter' | 'noChange' | 'allLabeled' = 'maxIter';
    constructor(props: SelfTrainingClassifierProps) {
        super();
        const { estimator, threshold = .75, criterion = 'threshold', kBest = 10, maxIter = 10 } = props ?? {} as SelfTrainingClassifierProps;
        if (!(estimator instanceof BaseEstimator) || typeof estimator.predictProba !== 'function') throw new Error('SelfTrainingClassifier requires a probabilistic classifier estimator');
        if ((criterion !== 'threshold' && criterion !== 'kBest') || !(threshold >= 0 && threshold < 1) || !Number.isInteger(kBest) || kBest < 1 || !Number.isInteger(maxIter) || maxIter < 0) throw new Error('invalid self-training parameters');
        this.estimator = estimator; this.threshold = threshold; this.criterion = criterion; this.kBest = kBest; this.maxIter = maxIter;
    }
    public getParams(): Params { return { estimator: this.estimator, threshold: this.threshold, criterion: this.criterion, kBest: this.kBest, maxIter: this.maxIter }; }
    public setParams(params: Params): this { const { own, nested } = splitEstimatorParams(params, this.constructor.name); const next = { ...this.getParams(), ...own }; const estimator = (next.estimator as BaseEstimator).clone(); if (Object.keys(nested).length > 0) estimator.setParams(nested); next.estimator = estimator; return super.setParams(next); }
    public fit(X: number[][], y: number[]): void {
        if (X.length === 0 || X.length !== y.length) throw new Error('X and y must be non-empty and have the same length');
        this.transductionState = y.slice();
        this.labeledIterationState = y.map(label => label === -1 ? -1 : 0);
        if (this.transductionState.every(label => label === -1)) throw new Error('at least one labeled sample is required');
        this.nIterState = 0;
        this.terminationConditionState = 'maxIter';
        this.fittedEstimator = undefined;
        if (this.transductionState.every(label => label !== -1)) this.terminationConditionState = 'allLabeled';
        for (let iteration = 1; this.terminationConditionState !== 'allLabeled' && iteration <= this.maxIter; iteration++) {
            this.nIterState = iteration;
            const labeled = this.transductionState.map((label, i) => ({ label, i })).filter(entry => entry.label !== -1);
            this.fittedEstimator = this.estimator.clone() as ProbabilisticClassifier;
            this.fittedEstimator.fit(labeled.map(entry => X[entry.i]), labeled.map(entry => entry.label));
            const unlabeled = this.transductionState.map((label, i) => ({ label, i })).filter(entry => entry.label === -1);
            if (unlabeled.length === 0) { this.terminationConditionState = 'allLabeled'; break; }
            const testX = unlabeled.map(entry => X[entry.i]);
            const probabilities = this.fittedEstimator.predictProba(testX);
            const predictions = this.fittedEstimator.predict(testX);
            let candidates = unlabeled.map((entry, local) => ({ index: entry.i, prediction: predictions[local], confidence: Math.max(...probabilities[local]) }));
            if (this.criterion === 'threshold') candidates = candidates.filter(entry => entry.confidence > this.threshold);
            else candidates = candidates.sort((a, b) => b.confidence - a.confidence || a.index - b.index).slice(0, Math.min(this.kBest, candidates.length));
            if (candidates.length === 0) { this.terminationConditionState = 'noChange'; break; }
            for (const candidate of candidates) { this.transductionState[candidate.index] = candidate.prediction; this.labeledIterationState[candidate.index] = iteration; }
            if (this.transductionState.every(label => label !== -1)) { this.terminationConditionState = 'allLabeled'; break; }
        }
        if (this.nIterState === this.maxIter) this.terminationConditionState = 'maxIter';
        if (this.transductionState.every(label => label !== -1)) this.terminationConditionState = 'allLabeled';
        const finalLabeled = this.transductionState.map((label, i) => ({ label, i })).filter(entry => entry.label !== -1);
        this.fittedEstimator = this.estimator.clone() as ProbabilisticClassifier;
        this.fittedEstimator.fit(finalLabeled.map(entry => X[entry.i]), finalLabeled.map(entry => entry.label));
    }
    private fitted(): ProbabilisticClassifier { if (!this.fittedEstimator) throw new Error('SelfTrainingClassifier is not fitted'); return this.fittedEstimator; }
    public predict(X: number[][]): number[] { return this.fitted().predict(X); }
    public predictProba(X: number[][]): number[][] { return this.fitted().predictProba(X); }
    public get transduction(): number[] { return this.transductionState.slice(); }
    public get labeledIteration(): number[] { return this.labeledIterationState.slice(); }
    public get nIter(): number { return this.nIterState; }
    public get terminationCondition(): string { return this.terminationConditionState; }
}
registerEstimator('SelfTrainingClassifier', SelfTrainingClassifier);
