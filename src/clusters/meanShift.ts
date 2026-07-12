import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';

export interface MeanShiftProps {
    /** kernel bandwidth (neighborhood radius used when shifting points) */
    bandwidth?: number;
    /** maximum number of shifting iterations */
    max_iter?: number;
    /** distance metric name */
    distanceType?: Distance.IDistanceType;
}

export class MeanShift extends ClusterBase {
    private bandwidth: number;
    private centers: number[][];
    private max_iter: number;
    private distanceType: Distance.IDistanceType;
    constructor(props?: MeanShiftProps);
    /** @deprecated positional form; prefer the props-object constructor */
    constructor(bandwidth?: number, max_iter?: number, distanceType?: Distance.IDistanceType);
    constructor(arg0: MeanShiftProps | number = {}, maxIterArg: number = 300, distanceTypeArg: Distance.IDistanceType = 'euclidean') {
        super();
        const props: MeanShiftProps = typeof arg0 === 'number'
            ? { bandwidth: arg0, max_iter: maxIterArg, distanceType: distanceTypeArg }
            : arg0;
        const { bandwidth = 1, max_iter = 300, distanceType = 'euclidean' } = props;
        this.bandwidth = bandwidth;
        this.centers = [];
        this.max_iter = max_iter;
        this.distanceType = distanceType;
        Distance.useDistance(distanceType); // validate the metric name eagerly
    }

    public getParams(): Params {
        return {
            bandwidth: this.bandwidth,
            max_iter: this.max_iter,
            distanceType: this.distanceType,
        };
    }

    /** resolved lazily from the metric name so instances stay JSON-serializable */
    private get distance(): Distance.IDistance {
        return Distance.useDistance(this.distanceType);
    }

    private shiftPoint(point: number[], samplesX: number[][]): number[] {
        const distance = this.distance;
        const neighbors = samplesX.filter(p => distance(p, point) <= this.bandwidth);
        if (neighbors.length === 0) return point;
        const dim = point.length;
        const mean = new Array(dim).fill(0);
        for (let i = 0; i < neighbors.length; i++) {
            for (let j = 0; j < dim; j++) {
                mean[j] += neighbors[i][j];
            }
        }
        for (let j = 0; j < dim; j++) {
            mean[j] /= neighbors.length;
        }
        return mean;
    }

    public fitPredict(samplesX: number[][]): number[] {
        let centers = samplesX.map(p => [...p]);
        for (let iter = 0; iter < this.max_iter; iter++) {
            let moved = false;
            for (let i = 0; i < centers.length; i++) {
                const newCenter = this.shiftPoint(centers[i], samplesX);
                for (let j = 0; j < newCenter.length; j++) {
                    if (Math.abs(newCenter[j] - centers[i][j]) > 1e-3) moved = true;
                    centers[i][j] = newCenter[j];
                }
            }
            if (!moved) break;
        }
        const distance = this.distance;
        const uniqueCenters: number[][] = [];
        const labels = new Array(samplesX.length);
        for (let i = 0; i < centers.length; i++) {
            let label = -1;
            for (let j = 0; j < uniqueCenters.length; j++) {
                if (distance(centers[i], uniqueCenters[j]) <= this.bandwidth / 2) {
                    label = j;
                    break;
                }
            }
            if (label === -1) {
                label = uniqueCenters.length;
                uniqueCenters.push(centers[i]);
            }
            labels[i] = label;
        }
        // assign each original sample to nearest center
        for (let i = 0; i < samplesX.length; i++) {
            let nearest = 0;
            let nearestDis = Infinity;
            for (let j = 0; j < uniqueCenters.length; j++) {
                const dis = distance(samplesX[i], uniqueCenters[j]);
                if (dis < nearestDis) {
                    nearestDis = dis;
                    nearest = j;
                }
            }
            labels[i] = nearest;
        }
        this.centers = uniqueCenters;
        return labels;
    }

    public getCentroids() {
        return this.centers;
    }
}
registerEstimator('MeanShift', MeanShift);
