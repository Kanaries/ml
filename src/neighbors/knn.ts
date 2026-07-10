import { ClassifierBase } from '../base';
import { Distance } from '../metrics';
import { getNeighborHits, IWeightType, resolveDistanceWeights, validateFitData, validatePredictData, weightedMode } from './utils';

export class KNearestNeighbors extends ClassifierBase {
    private trainX: number[][];
    private trainY: number[];
    private metric: Distance.IDistanceType;
    private kNeighbors: number;
    private weightType: IWeightType;
    private pNorm: number;
    private fitted: boolean;
    public constructor(
        kNeighbors: number = 5,
        weightType: IWeightType = 'uniform',
        distanceType: Distance.IDistanceType = 'euclidean',
        pNorm: number = 2
    ) {
        super();
        if (!Number.isInteger(kNeighbors) || kNeighbors <= 0) {
            throw new Error('kNeighbors must be an integer > 0');
        }
        if (weightType !== 'uniform' && weightType !== 'distance') {
            throw new Error(`Do not support weightType: ${weightType}. Use 'uniform' or 'distance' instead.`);
        }
        this.trainX = [];
        this.trainY = [];
        this.metric = distanceType;
        this.kNeighbors = kNeighbors;
        this.weightType = weightType;
        this.pNorm = pNorm;
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
            const hits = getNeighborHits(this.trainX, sample, this.metric, this.pNorm).slice(0, Math.min(this.kNeighbors, this.trainX.length));
            const labels = hits.map(hit => this.trainY[hit.index]);
            const weights = this.weightType === 'uniform'
                ? new Array(labels.length).fill(1)
                : resolveDistanceWeights(hits.map(hit => hit.distance));
            return weightedMode(labels, weights);
        });
    }
}

/** @deprecated Use KNearestNeighbors instead. */
export const KNearstNeighbors = KNearestNeighbors;
/** @deprecated Use KNearestNeighbors instead. */
export type KNearstNeighbors = KNearestNeighbors;
