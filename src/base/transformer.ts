import { BaseEstimator } from "./estimator";
import type { FeatureData, FeatureDataKind } from '../data';

/**
 * Base class for feature transformers (scalers, encoders, decomposition,
 * feature selection, ...). `y` is accepted for pipeline compatibility and
 * used only by supervised transformers.
 */
export abstract class TransformerBase<TInput extends FeatureData = number[][], TOutput extends FeatureData = number[][]> extends BaseEstimator {
    /** Runtime companion to the generic input type (which is erased in JS). */
    public readonly acceptedInputKinds: readonly FeatureDataKind[] = ['dense'];
    public abstract fit(X: TInput, y?: number[]): void;
    public abstract transform(X: TInput): TOutput;
    public fitTransform(X: TInput, y?: number[]): TOutput {
        this.fit(X, y);
        return this.transform(X);
    }
    /** Optional capability: invert the transformation where well-defined. */
    public inverseTransform?(X: TOutput): TInput;
}
