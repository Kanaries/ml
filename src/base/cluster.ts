import { accuracyScore } from '../metrics';

export abstract class ClusterBase {
    public abstract fitPredict(trainX: number[][], trainY: number[]): void;
}
