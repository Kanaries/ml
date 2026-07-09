import { XGBoostBase, XGBoostProps } from './xgboostBase';

export interface XGBoostRegressorProps extends XGBoostProps {}

/**
 * XGBoost with the reg:squarederror objective: g = F - y, h = 1.
 * The default base_score = 0.5 matches the xgboost library.
 */
export class XGBoostRegressor extends XGBoostBase {
    constructor(props: XGBoostRegressorProps = {}) {
        super(props);
    }

    protected gradients(F: number[], y: number[]): { g: number[]; h: number[] } {
        return {
            g: F.map((f, i) => f - y[i]),
            h: new Array(F.length).fill(1),
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.validateFitInput(trainX, trainY);
        this.fitBoosted(trainX, trainY, this.baseScore);
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return this.predictMargin(testX);
    }
}
