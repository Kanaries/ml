import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';
import { getNeighborHits, IWeightType, resolveDistanceWeights, validateFitData, validatePredictData, weightedMode } from './utils';

export interface KNearestNeighborsProps {
    kNeighbors?: number;
    weightType?: IWeightType;
    distanceType?: Distance.IDistanceType;
    pNorm?: number;
}

export class KNearestNeighbors extends ClassifierBase {
    private trainX: number[][];
    private trainY: number[];
    private metric: Distance.IDistanceType;
    private kNeighbors: number;
    private weightType: IWeightType;
    private pNorm: number;
    private fitted: boolean;
    public constructor(props?: KNearestNeighborsProps);
    /** @deprecated positional form; prefer the props-object constructor */
    public constructor(kNeighbors?: number, weightType?: IWeightType, distanceType?: Distance.IDistanceType, pNorm?: number);
    public constructor(
        arg0: KNearestNeighborsProps | number = {},
        weightTypeArg: IWeightType = 'uniform',
        distanceTypeArg: Distance.IDistanceType = 'euclidean',
        pNormArg: number = 2
    ) {
        super();
        const props: KNearestNeighborsProps = typeof arg0 === 'number'
            ? { kNeighbors: arg0, weightType: weightTypeArg, distanceType: distanceTypeArg, pNorm: pNormArg }
            : arg0;
        const {
            kNeighbors = 5,
            weightType = 'uniform',
            distanceType = 'euclidean',
            pNorm = 2,
        } = props;
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
    public getParams(): Params {
        return {
            kNeighbors: this.kNeighbors,
            weightType: this.weightType,
            distanceType: this.metric,
            pNorm: this.pNorm,
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
            const hits = getNeighborHits(this.trainX, sample, this.metric, this.pNorm).slice(0, Math.min(this.kNeighbors, this.trainX.length));
            const labels = hits.map(hit => this.trainY[hit.index]);
            const weights = this.weightType === 'uniform'
                ? new Array(labels.length).fill(1)
                : resolveDistanceWeights(hits.map(hit => hit.distance));
            return weightedMode(labels, weights);
        });
    }
}
registerEstimator('KNearestNeighbors', KNearestNeighbors);

/** @deprecated Use KNearestNeighbors instead. */
export const KNearstNeighbors = KNearestNeighbors;
/** @deprecated Use KNearestNeighbors instead. */
export type KNearstNeighbors = KNearestNeighbors;
