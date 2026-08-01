import { Params, registerEstimator } from '../base/estimator';
import { KFold, isClassifierLike, resolveScoring, ScoringFunction, SplitterLike, StratifiedKFold } from '../utils/modelSelection';
import { FeatureSelectingEstimator, resolveFeatureCount, resolveStep } from './common';
import { RFE, RFEProps } from './rfe';

export interface RFECVProps extends Omit<RFEProps, 'nFeaturesToSelect'> {
    minFeaturesToSelect?: number;
    cv?: number | SplitterLike;
    scoring?: ScoringFunction | string;
}

export class RFECV extends RFE {
    private minFeaturesToSelect: number;
    private cv: number | SplitterLike;
    private scoring?: ScoringFunction | string;
    private cvScores: number[] = [];
    private featureCounts: number[] = [];

    constructor(props: RFECVProps) {
        super({ estimator: props.estimator, step: props.step });
        this.minFeaturesToSelect = props.minFeaturesToSelect ?? 1;
        this.cv = props.cv ?? 5;
        this.scoring = props.scoring;
    }

    public getParams(): Params {
        return { estimator: this.estimator, step: this.step, minFeaturesToSelect: this.minFeaturesToSelect, cv: this.cv, scoring: this.scoring };
    }

    public fit(X: number[][], y?: number[]): void {
        if (!y) throw new Error('RFECV requires y');
        const min = resolveFeatureCount(this.minFeaturesToSelect, X[0].length, 1);
        const splitter = typeof this.cv === 'number'
            ? isClassifierLike(this.estimator)
                ? new StratifiedKFold({ nSplits: this.cv })
                : new KFold({ nSplits: this.cv })
            : this.cv;
        const folds = splitter.split(X, y);
        let bestCount = min;
        let bestScore = -Infinity;
        this.cvScores = [];
        this.featureCounts = [X[0].length];
        let remaining = X[0].length;
        const fixedStep = resolveStep(this.step, X[0].length);
        while (remaining > min) {
            remaining -= Math.min(fixedStep, remaining - min);
            this.featureCounts.push(remaining);
        }
        this.featureCounts.sort((a, b) => a - b);
        const scoring = resolveScoring(this.scoring);
        for (const count of this.featureCounts) {
            let total = 0;
            for (const fold of folds) {
                const selector = new RFE({ estimator: this.estimator, nFeaturesToSelect: count, step: this.step });
                const trainX = fold.trainIndices.map(i => X[i]);
                const trainY = fold.trainIndices.map(i => y[i]);
                selector.fit(trainX, trainY);
                const testX = fold.testIndices.map(i => X[i]);
                const testY = fold.testIndices.map(i => y[i]);
                total += scoring ? scoring(selector.predict(testX), testY) : selector.score(testX, testY);
            }
            const score = total / folds.length;
            this.cvScores.push(score);
            if (score > bestScore) {
                bestScore = score;
                bestCount = count;
            }
        }
        this.nFeaturesToSelect = bestCount;
        super.fit(X, y);
    }

    public get gridScores(): number[] { return this.cvScores.slice(); }
    public get cvResults(): { nFeatures: number[]; meanTestScore: number[] } {
        return { nFeatures: this.featureCounts.slice(), meanTestScore: this.cvScores.slice() };
    }
}
registerEstimator('RFECV', RFECV);
