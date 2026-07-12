import { BaseEstimator } from "./estimator";

/**
 * Base class for feature transformers (scalers, encoders, decomposition,
 * feature selection, ...). `y` is accepted for pipeline compatibility and
 * used only by supervised transformers.
 */
export abstract class TransformerBase extends BaseEstimator {
    public abstract fit(X: number[][], y?: number[]): void;
    public abstract transform(X: number[][]): number[][];
    public fitTransform(X: number[][], y?: number[]): number[][] {
        this.fit(X, y);
        return this.transform(X);
    }
    /** Optional capability: invert the transformation where well-defined. */
    public inverseTransform?(X: number[][]): number[][];
}
