import { RegressorBase } from '../../base';
import { registerEstimator, Params } from '../../base/estimator';
import {
    XGBoostParams,
    XGBoostProps,
    fitBoostedTrees,
    predictBoostedMargin,
    resolveXGBoostProps,
    validateXGBoostFitInput,
} from './xgboostBase';
import { XGBTree } from './xgbTree';
import { averageImportances } from '../../tree/featureImportances';

export interface XGBoostRegressorProps extends XGBoostProps {}

/** reg:squarederror objective: g = F - y, h = 1. */
function squaredErrorGradients(F: number[], y: number[]): { g: number[]; h: number[] } {
    return {
        g: F.map((f, i) => f - y[i]),
        h: new Array(F.length).fill(1),
    };
}

/**
 * XGBoost with the reg:squarederror objective.
 * The default base_score = 0.5 matches the xgboost library.
 */
export class XGBoostRegressor extends RegressorBase {
    private params: XGBoostParams;
    private trees: XGBTree[];
    private baseMargin: number;
    private fitted: boolean;
    private nFeatures: number;

    constructor(props: XGBoostRegressorProps = {}) {
        super();
        this.params = resolveXGBoostProps(props);
        this.trees = [];
        this.baseMargin = 0;
        this.fitted = false;
        this.nFeatures = 0;
    }

    public getParams(): Params {
        return { ...this.params };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateXGBoostFitInput(trainX, trainY);
        this.nFeatures = trainX[0].length;
        this.baseMargin = this.params.baseScore;
        this.trees = fitBoostedTrees(trainX, trainY, this.baseMargin, this.params, squaredErrorGradients);
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return predictBoostedMargin(this.trees, this.baseMargin, this.params.learningRate, testX);
    }
    public get featureImportances(): number[] {
        if (!this.fitted) throw new Error('model is not fitted');
        if (!Number.isInteger(this.nFeatures) || this.nFeatures <= 0) {
            throw new Error('featureImportances are unavailable for this legacy XGBoost model; refit the model');
        }
        const raw = this.trees.map(tree => tree.featureImportances(this.nFeatures));
        return averageImportances(raw);
    }
}
registerEstimator('XGBoostRegressor', XGBoostRegressor);
