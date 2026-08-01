import { BaseEstimator, RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { LinearRegression } from '../linear/linearRegression';
import { ElementwiseFunc, FunctionTransformer } from '../utils/preprocessingExtra';

interface RegressorLike extends BaseEstimator { fit(X: number[][], y: number[]): void; predict(X: number[][]): number[]; }
interface TargetTransformer extends BaseEstimator { fit(X: number[][]): void; transform(X: number[][]): number[][]; inverseTransform?(X: number[][]): number[][]; }
export interface TransformedTargetRegressorProps { regressor?: RegressorLike; transformer?: TargetTransformer | null; func?: ElementwiseFunc | string; inverseFunc?: ElementwiseFunc | string; }

/** Fit a regressor on a transformed target, then invert predictions to the original target space. */
export class TransformedTargetRegressor extends RegressorBase {
    private regressor: RegressorLike;
    private transformer: TargetTransformer | null;
    private func: ElementwiseFunc | string;
    private inverseFunc: ElementwiseFunc | string;
    private fittedRegressorState?: RegressorLike;
    private fittedTransformerState?: TargetTransformer;

    constructor(props: TransformedTargetRegressorProps = {}) {
        super();
        if ((props.func === undefined) !== (props.inverseFunc === undefined)) throw new Error('func and inverseFunc must be provided together');
        this.regressor = props.regressor ?? new LinearRegression();
        this.transformer = props.transformer ?? null;
        this.func = props.func ?? 'identity'; this.inverseFunc = props.inverseFunc ?? 'identity';
        if (this.transformer && (props.func !== undefined || props.inverseFunc !== undefined)) throw new Error('transformer and func/inverseFunc are mutually exclusive');
        if (typeof this.regressor.fit !== 'function' || typeof this.regressor.predict !== 'function') throw new Error('regressor must implement fit and predict');
        if (this.transformer && (typeof this.transformer.fit !== 'function' || typeof this.transformer.transform !== 'function' || typeof this.transformer.inverseTransform !== 'function')) throw new Error('transformer must implement fit, transform, and inverseTransform');
    }
    public getParams(): Params { return { regressor: this.regressor, transformer: this.transformer, func: this.transformer ? undefined : this.func, inverseFunc: this.transformer ? undefined : this.inverseFunc }; }
    public setParams(params: Params): this {
        const own: Params = {}; const regressorParams: Params = {}; const transformerParams: Params = {};
        for (const [key, value] of Object.entries(params)) {
            if (key.startsWith('regressor__')) regressorParams[key.slice(11)] = value;
            else if (key.startsWith('transformer__')) transformerParams[key.slice(13)] = value;
            else own[key] = value;
        }
        if ((Object.prototype.hasOwnProperty.call(own, 'func')) !== (Object.prototype.hasOwnProperty.call(own, 'inverseFunc'))) throw new Error('func and inverseFunc must be set together');
        const next = { ...this.getParams(), ...own } as unknown as TransformedTargetRegressorProps & { regressor: RegressorLike };
        if (own.transformer) { next.func = undefined; next.inverseFunc = undefined; }
        if ((own.func !== undefined || own.inverseFunc !== undefined) && own.transformer === undefined) next.transformer = null;
        next.regressor = next.regressor.clone(); if (next.transformer) next.transformer = next.transformer.clone() as TargetTransformer;
        if (Object.keys(regressorParams).length) next.regressor.setParams(regressorParams);
        if (Object.keys(transformerParams).length) { if (!next.transformer) throw new Error('transformer__ parameters require a transformer'); next.transformer.setParams(transformerParams); }
        return super.setParams(next as unknown as Params);
    }
    public fit(X: number[][], y: number[]): void {
        if (X.length === 0 || X.length !== y.length || y.some(value => !Number.isFinite(value))) throw new Error('X and y must be non-empty, aligned, and finite');
        const transformer = this.transformer ? this.transformer.clone() as TargetTransformer : new FunctionTransformer({ func: this.func, inverseFunc: this.inverseFunc });
        const target = y.map(value => [value]);
        transformer.fit(target);
        const transformed = transformer.transform(target);
        if (transformed.some(row => row.length !== 1 || !Number.isFinite(row[0]))) throw new Error('target transformer must preserve one target column');
        const regressor = this.regressor.clone() as RegressorLike;
        regressor.fit(X, transformed.map(row => row[0]));
        this.fittedTransformerState = transformer; this.fittedRegressorState = regressor;
    }
    public predict(X: number[][]): number[] {
        if (!this.fittedRegressorState || !this.fittedTransformerState) throw new Error('TransformedTargetRegressor is not fitted');
        const values = this.fittedRegressorState.predict(X).map(value => [value]);
        return this.fittedTransformerState.inverseTransform!(values).map(row => row[0]);
    }
}
registerEstimator('TransformedTargetRegressor', TransformedTargetRegressor);
