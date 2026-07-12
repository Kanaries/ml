import { BaseEstimator } from './estimator';

export abstract class OutlierBase extends BaseEstimator {
    public abstract fit (samplesX: number[][]): void;
    public abstract predict(samplesX: number[][]): number[];
    public fitPredict (samplesX: number[][]): number[] {
        this.fit(samplesX);
        return this.predict(samplesX);
    }
}
