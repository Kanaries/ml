import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';
import { getRadiusHits, resolveDistanceWeights, validateFitData, validatePredictData, weightedMode } from './utils';

export interface RadiusNeighborsClassifierProps {
    radius?: number;
    weights?: 'uniform' | 'distance';
    metric?: Distance.IDistanceType;
    p?: number;
    outlierLabel?: number | null;
}

export class RadiusNeighborsClassifier extends ClassifierBase {
    private radius: number;
    private weights: 'uniform' | 'distance';
    private metric: Distance.IDistanceType;
    private p: number;
    private trainX: number[][];
    private trainY: number[];
    private fitted: boolean;
    private outlierLabel: number | null;

    constructor(props: RadiusNeighborsClassifierProps = {}) {
        super();
        const { radius = 1, weights = 'uniform', metric = 'euclidean', p = 2, outlierLabel = null } = props;
        if (!Number.isFinite(radius) || radius < 0) {
            throw new Error('radius must be a finite number >= 0');
        }
        this.radius = radius;
        this.weights = weights;
        this.metric = metric;
        this.p = p;
        this.outlierLabel = outlierLabel;
        this.trainX = [];
        this.trainY = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            radius: this.radius,
            weights: this.weights,
            metric: this.metric,
            p: this.p,
            outlierLabel: this.outlierLabel,
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
            const hits = getRadiusHits(this.trainX, sample, this.radius, this.metric, this.p);
            if (hits.length === 0) {
                if (this.outlierLabel !== null) {
                    return this.outlierLabel;
                }
                throw new Error('No neighbors found within the configured radius');
            }
            const labels = hits.map(hit => this.trainY[hit.index]);
            if (this.weights === 'uniform') {
                const weights = new Array(labels.length).fill(1);
                return weightedMode(labels, weights);
            }
            const weights = resolveDistanceWeights(hits.map(hit => hit.distance));
            return weightedMode(labels, weights);
        });
    }
}
registerEstimator('RadiusNeighborsClassifier', RadiusNeighborsClassifier);
