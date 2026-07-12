import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import {
    RegressionLossName,
    SGDLearningRate,
    SGDPenalty,
    getRegressionLoss,
    plainSGD,
    validatePenalty,
    validateSGDData,
    validateSchedule,
} from './sgdBase';

export interface SGDRegressorProps {
    loss?: RegressionLossName;
    /** Threshold of the huber / (squared) epsilon-insensitive losses. */
    epsilon?: number;
    penalty?: SGDPenalty;
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    tol?: number | null;
    shuffle?: boolean;
    randomState?: number;
    learningRate?: SGDLearningRate;
    eta0?: number;
    powerT?: number;
    nIterNoChange?: number;
}

/**
 * Linear regressor trained with plain (non-averaged) stochastic gradient
 * descent, mirroring scikit-learn's `SGDRegressor` (default schedule
 * `invscaling` with eta0=0.01, powerT=0.25).
 */
export class SGDRegressor extends RegressorBase {
    private loss: RegressionLossName;
    private epsilon: number;
    private penalty: SGDPenalty;
    private alpha: number;
    private l1Ratio: number;
    private fitIntercept: boolean;
    private maxIter: number;
    private tol: number | null;
    private shuffle: boolean;
    private randomState?: number;
    private learningRate: SGDLearningRate;
    private eta0: number;
    private powerT: number;
    private nIterNoChange: number;

    private coef: number[];
    private intercept: number;
    private fitted: boolean;
    private nIter: number;

    constructor(props: SGDRegressorProps = {}) {
        super();
        this.loss = props.loss ?? 'squaredError';
        this.epsilon = props.epsilon ?? 0.1;
        this.penalty = validatePenalty(props.penalty === undefined ? 'l2' : props.penalty);
        this.alpha = props.alpha ?? 1e-4;
        this.l1Ratio = props.l1Ratio ?? 0.15;
        this.fitIntercept = props.fitIntercept ?? true;
        this.maxIter = props.maxIter ?? 1000;
        this.tol = props.tol === undefined ? 1e-3 : props.tol;
        this.shuffle = props.shuffle ?? true;
        this.randomState = props.randomState;
        this.learningRate = props.learningRate ?? 'invscaling';
        this.eta0 = props.eta0 ?? 0.01;
        this.powerT = props.powerT ?? 0.25;
        this.nIterNoChange = props.nIterNoChange ?? 5;
        this.coef = [];
        this.intercept = 0;
        this.fitted = false;
        this.nIter = 0;
    }

    public getParams(): Params {
        return {
            loss: this.loss,
            epsilon: this.epsilon,
            penalty: this.penalty,
            alpha: this.alpha,
            l1Ratio: this.l1Ratio,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tol: this.tol,
            shuffle: this.shuffle,
            randomState: this.randomState,
            learningRate: this.learningRate,
            eta0: this.eta0,
            powerT: this.powerT,
            nIterNoChange: this.nIterNoChange,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateSGDData(trainX, trainY);
        const loss = getRegressionLoss(this.loss, this.epsilon);
        validateSchedule(this.learningRate, this.eta0, this.alpha);
        const rand = createRandomGenerator(this.randomState);
        const { weights, intercept, nIter } = plainSGD({
            X: trainX,
            y: trainY,
            loss,
            penalty: this.penalty,
            alpha: this.alpha,
            l1Ratio: this.l1Ratio,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tol: this.tol,
            shuffle: this.shuffle,
            rand,
            learningRate: this.learningRate,
            eta0: this.eta0,
            powerT: this.powerT,
            nIterNoChange: this.nIterNoChange,
        });
        this.coef = weights;
        this.intercept = intercept;
        this.nIter = nIter;
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('SGDRegressor must be fitted before calling predict');
        }
        return testX.map(x => {
            let p = this.intercept;
            for (let j = 0; j < this.coef.length; j++) p += this.coef[j] * x[j];
            return p;
        });
    }

    public getCoef(): number[] {
        return this.coef.slice();
    }

    public getIntercept(): number {
        return this.intercept;
    }

    /** Number of epochs run in the last fit. */
    public getNIter(): number {
        return this.nIter;
    }
}
registerEstimator('SGDRegressor', SGDRegressor);
