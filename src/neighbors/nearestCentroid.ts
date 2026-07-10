import { Distance } from '../metrics';
import { validateFitData, validatePredictData } from './utils';

export interface NearestCentroidProps {
    metric?: Distance.IDistanceType;
    p?: number;
}

function median(values: number[]): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
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

        const grouped: number[][][] = this.classes.map(() => []);
        for (let i = 0; i < trainX.length; i++) {
            const cls = classIndex.get(trainY[i]);
            if (cls === undefined) {
                continue;
            }
            grouped[cls].push(trainX[i]);
        }

        // sklearn semantics: manhattan uses the per-feature median as the
        // centroid; other metrics use the per-feature mean.
        const useMedian = this.metric === 'manhattan' || this.metric === '1-norm';
        this.centroids = grouped.map(rows =>
            Array.from({ length: nFeatures }, (_, j) => {
                const column = rows.map(row => row[j]);
                if (useMedian) {
                    return median(column);
                }
                return column.reduce((sum, value) => sum + value, 0) / column.length;
            })
        );
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
