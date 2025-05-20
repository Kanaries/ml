import { ClusterBase } from '../base/cluster';
import { Distance } from '../metrics';

export class MeanShift extends ClusterBase {
    private bandwidth: number;
    private centers: number[][];
    private max_iter: number;
    private distance: Distance.IDistance;
    public constructor(bandwidth: number = 1, max_iter: number = 300, distanceType: Distance.IDistanceType = 'euclidiean') {
        super();
        this.bandwidth = bandwidth;
        this.centers = [];
        this.max_iter = max_iter;
        this.distance = Distance.useDistance(distanceType);
    }

    private shiftPoint(point: number[], samplesX: number[][]): number[] {
        const neighbors = samplesX.filter(p => this.distance(p, point) <= this.bandwidth);
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
        const uniqueCenters: number[][] = [];
        const labels = new Array(samplesX.length);
        for (let i = 0; i < centers.length; i++) {
            let label = -1;
            for (let j = 0; j < uniqueCenters.length; j++) {
                if (this.distance(centers[i], uniqueCenters[j]) <= this.bandwidth / 2) {
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
                const dis = this.distance(samplesX[i], uniqueCenters[j]);
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
