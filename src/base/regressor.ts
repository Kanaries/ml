import { r2Score } from "../metrics";
import { BaseEstimator } from "./estimator";

export abstract class RegressorBase extends BaseEstimator {
    public abstract fit(trainX: number[][], trainY: number[], sampleWeight?: number[]): void;
    public abstract predict(testX: number[][]): number[];
    public score (X: number[][], Y: number[]): number {
        return r2Score(this.predict(X), Y);
    }
}
