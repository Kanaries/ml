import { Distance } from '../metrics';
import { getRadiusHits, resolveDistanceWeights, validateFitData, validatePredictData, weightedAverage } from './utils';

export interface RadiusNeighborsRegressorProps {
    radius?: number;
    weights?: 'uniform' | 'distance';
    metric?: Distance.IDistanceType;
    p?: number;
}

export class RadiusNeighborsRegressor {
    private radius: number;
    private weights: 'uniform' | 'distance';
    private metric: Distance.IDistanceType;
    private p: number;
    private trainX: number[][];
    private trainY: number[];
    private fitted: boolean;

    constructor(props: RadiusNeighborsRegressorProps = {}) {
        const { radius = 1, weights = 'uniform', metric = 'euclidean', p = 2 } = props;
        if (!Number.isFinite(radius) || radius < 0) {
            throw new Error('radius must be a finite number >= 0');
        }
        this.radius = radius;
        this.weights = weights;
        this.metric = metric;
        this.p = p;
        this.trainX = [];
        this.trainY = [];
        this.fitted = false;
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
            const hits = getRadiusHits(this.trainX, sample, this.radius, this.metric, this.p);
            if (hits.length === 0) {
                return Number.NaN;
            }
            const values = hits.map(hit => this.trainY[hit.index]);
            if (this.weights === 'uniform') {
                return values.reduce((sum, value) => sum + value, 0) / values.length;
            }
            const weights = resolveDistanceWeights(hits.map(hit => hit.distance));
            return weightedAverage(values, weights);
        });
    }
}
