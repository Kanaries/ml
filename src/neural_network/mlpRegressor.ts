import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { r2Score } from '../metrics';
import {
    MLPParams,
    MLPProps,
    MLPWeights,
    predictMLPOutput,
    resolveMLPProps,
    trainMLP,
    validateMLPFitInput,
} from './mlpBase';

export interface MLPRegressorProps extends MLPProps {}

/**
 * Multi-layer perceptron regressor trained with minibatch backpropagation
 * (sklearn `MLPRegressor` parity for the 'adam' and 'sgd' solvers; the
 * lbfgs solver is not implemented).
 *
 * Identity output activation with squared loss (sklearn's
 * `squared_loss = mean((y - p)^2) / 2`); single target `y: number[]` only.
 * The L2 penalty `alpha` applies to weights only, never biases.
 *
 * Early stopping (`earlyStopping: true`) holds out the LAST
 * `validationFraction` of the rows as a plain deterministic split (sklearn
 * shuffles via train_test_split — documented deviation) and restores the
 * weights with the best validation R².
 */
export class MLPRegressor extends RegressorBase {
    private params: MLPParams;
    private fitted: boolean;
    private coefsState: number[][][];
    private interceptsState: number[][];
    private lossCurveState: number[];
    private nIterState: number;

    constructor(props: MLPRegressorProps = {}) {
        super();
        this.params = resolveMLPProps(props);
        this.fitted = false;
        this.coefsState = [];
        this.interceptsState = [];
        this.lossCurveState = [];
        this.nIterState = 0;
    }

    public getParams(): Params {
        return { ...this.params, hiddenLayerSizes: [...this.params.hiddenLayerSizes] };
    }

    /** Mean training loss recorded at the end of each epoch. */
    public get lossCurve(): number[] {
        return [...this.lossCurveState];
    }

    /** Number of epochs actually run by the last fit. */
    public get nIter(): number {
        return this.nIterState;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateMLPFitInput(trainX, trainY);
        for (const v of trainY) {
            if (!Number.isFinite(v)) {
                throw new Error('y contains NaN or non-finite values, which are not supported');
            }
        }
        const Y = trainY.map(v => [v]);
        const result = trainMLP(trainX, Y, this.params, {
            outActivation: 'identity',
            lossKind: 'squared',
            validationScore: (weights: MLPWeights, XVal: number[][], YVal: number[][]) => {
                const out = predictMLPOutput(XVal, weights, this.params.activation, 'identity');
                return r2Score(out.map(row => row[0]), YVal.map(row => row[0]));
            },
        });
        this.coefsState = result.weights.coefs;
        this.interceptsState = result.weights.intercepts;
        this.lossCurveState = result.lossCurve;
        this.nIterState = result.nIter;
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        const weights: MLPWeights = { coefs: this.coefsState, intercepts: this.interceptsState };
        return predictMLPOutput(testX, weights, this.params.activation, 'identity').map(row => row[0]);
    }
}
registerEstimator('MLPRegressor', MLPRegressor);
