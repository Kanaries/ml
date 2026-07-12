import { accuracyScore } from "../metrics";
import { BaseEstimator } from "./estimator";

export abstract class ClassifierBase extends BaseEstimator {
    public abstract fit(trainX: number[][], trainY: number[], sampleWeight?: number[]): void;
    public abstract predict(testX: number[][]): number[];
    /**
     * Class-membership probabilities, shape [nSamples][nClasses], columns
     * ordered by the estimator's sorted `classes`. Optional capability —
     * implemented by estimators with a meaningful probability model.
     */
    public predictProba?(testX: number[][]): number[][];
    /** Signed distance to the decision boundary (optional capability). */
    public decisionFunction?(testX: number[][]): number[] | number[][];
    public score (X: number[][], Y: number[]): number {
        const resultY = this.predict(X);
        return accuracyScore(resultY, Y);
    }
}
