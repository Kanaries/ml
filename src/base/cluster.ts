import { BaseEstimator } from './estimator';

export abstract class ClusterBase extends BaseEstimator {
    public abstract fitPredict(trainX: number[][], sampleWeights?: number[]): number[];
}
