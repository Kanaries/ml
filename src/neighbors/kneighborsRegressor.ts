import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';
import { getNeighborHits, resolveDistanceWeights, validateFitData, validatePredictData, weightedAverage } from './utils';

export interface KNeighborsRegressorProps {
    nNeighbors?: number;
    weights?: 'uniform' | 'distance';
    metric?: Distance.IDistanceType;
    p?: number;
}

export class KNeighborsRegressor extends RegressorBase {
    private nNeighbors: number;
    private weights: 'uniform' | 'distance';
    private metric: Distance.IDistanceType;
    private p: number;
    private trainX: number[][];
    private trainY: number[];
    private fitted: boolean;

    constructor(props?: KNeighborsRegressorProps);
    /** @deprecated positional form; prefer the props-object constructor */
    constructor(nNeighbors: number, weights?: 'uniform' | 'distance', metric?: Distance.IDistanceType, p?: number);
    constructor(
        propsOrNeighbors: KNeighborsRegressorProps | number = {},
        weightsArg: 'uniform' | 'distance' = 'uniform',
        metricArg: Distance.IDistanceType = 'euclidean',
        pArg: number = 2,
    ) {
        super();
        const props = typeof propsOrNeighbors === 'number'
            ? { nNeighbors: propsOrNeighbors, weights: weightsArg, metric: metricArg, p: pArg }
            : propsOrNeighbors;
        const { nNeighbors = 5, weights = 'uniform', metric = 'euclidean', p = 2 } = props;
        if (!Number.isInteger(nNeighbors) || nNeighbors <= 0) {
            throw new Error('nNeighbors must be an integer > 0');
        }
        this.nNeighbors = nNeighbors;
        this.weights = weights;
        this.metric = metric;
        this.p = p;
        this.trainX = [];
        this.trainY = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            nNeighbors: this.nNeighbors,
            weights: this.weights,
            metric: this.metric,
            p: this.p,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateFitData(trainX, trainY);
        this.trainX = trainX;
        this.trainY = trainY;
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        validatePredictData(testX, this.trainX[0].length);
        return testX.map(sample => {
            const hits = getNeighborHits(this.trainX, sample, this.metric, this.p).slice(0, Math.min(this.nNeighbors, this.trainX.length));
            const values = hits.map(hit => this.trainY[hit.index]);
            if (this.weights === 'uniform') {
                return values.reduce((sum, value) => sum + value, 0) / values.length;
            }
            const weights = resolveDistanceWeights(hits.map(hit => hit.distance));
            return weightedAverage(values, weights);
        });
    }
}
registerEstimator('KNeighborsRegressor', KNeighborsRegressor);
