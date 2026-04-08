import { Distance } from '../metrics';
import { validateFitData, validatePredictData } from './utils';

export interface NearestCentroidProps {
    metric?: Distance.IDistanceType;
    p?: number;
}

export class NearestCentroid {
    private metric: Distance.IDistanceType;
    private p: number;
    private classes: number[];
    private centroids: number[][];
    private fitted: boolean;

    constructor(props: NearestCentroidProps = {}) {
        const { metric = 'euclidean', p = 2 } = props;
        this.metric = metric;
        this.p = p;
        this.classes = [];
        this.centroids = [];
        this.fitted = false;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateFitData(trainX, trainY);
        const nFeatures = trainX[0].length;
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        const classIndex = new Map<number, number>();
        this.classes.forEach((label, index) => classIndex.set(label, index));

        const sums = Array.from({ length: this.classes.length }, () => new Array(nFeatures).fill(0));
        const counts = new Array(this.classes.length).fill(0);

        for (let i = 0; i < trainX.length; i++) {
            const cls = classIndex.get(trainY[i]);
            if (cls === undefined) {
                continue;
            }
            counts[cls] += 1;
            for (let j = 0; j < nFeatures; j++) {
                sums[cls][j] += trainX[i][j];
            }
        }

        this.centroids = sums.map((row, classIdx) => row.map(value => value / counts[classIdx]));
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        validatePredictData(testX, this.centroids[0].length);
        const distance = Distance.useDistance(this.metric);
        return testX.map(sample => {
            let bestClass = this.classes[0];
            let bestDistance = Infinity;
            for (let i = 0; i < this.centroids.length; i++) {
                const centroidDistance = distance(sample, this.centroids[i], this.p);
                const label = this.classes[i];
                if (centroidDistance < bestDistance || (centroidDistance === bestDistance && label < bestClass)) {
                    bestDistance = centroidDistance;
                    bestClass = label;
                }
            }
            return bestClass;
        });
    }
}
