import { ClusterBase } from '../base/cluster';
import { Distance } from '../metrics';

export interface OPTICSOptions {
    min_samples?: number;
    max_eps?: number;
    metric?: Distance.IDistanceType;
    p?: number;
    eps?: number;
}

export class OPTICS extends ClusterBase {
    private minSamples: number;
    private maxEps: number;
    private distance: Distance.IDistance;
    private p: number;
    private eps: number;
    private reachability: number[] = [];
    private coreDistances: number[] = [];
    private ordering: number[] = [];

    constructor(options: OPTICSOptions = {}) {
        super();
        const { min_samples = 5, max_eps, metric = 'euclidean', p = 2, eps = 0.5 } = options;
        this.minSamples = min_samples;
        this.eps = eps;
        this.maxEps = max_eps !== undefined ? max_eps : eps;
        this.distance = Distance.useDistance(metric);
        this.p = p;
    }

    public fitPredict(samplesX: number[][]): number[] {
        const n = samplesX.length;
        this.reachability = new Array(n).fill(Infinity);
        this.coreDistances = new Array(n).fill(Infinity);
        this.ordering = [];

        const distanceMatrix: number[][] = [];
        for (let i = 0; i < n; i++) {
            distanceMatrix[i] = [];
            for (let j = 0; j < n; j++) {
                distanceMatrix[i][j] = this.distance(samplesX[i], samplesX[j], this.p);
            }
        }

        for (let i = 0; i < n; i++) {
            const sorted = distanceMatrix[i].slice().sort((a, b) => a - b);
            if (this.minSamples <= sorted.length) {
                this.coreDistances[i] = sorted[this.minSamples - 1];
            } else {
                this.coreDistances[i] = Infinity;
            }
        }

        const processed = new Array(n).fill(false);
        for (let i = 0; i < n; i++) {
            if (!processed[i]) {
                this.expandClusterOrder(i, processed, distanceMatrix);
            }
        }

        const labels = new Array(n).fill(-1);
        let clusterId = -1;
        for (const index of this.ordering) {
            if (this.reachability[index] > this.eps) {
                if (this.coreDistances[index] <= this.eps) {
                    clusterId += 1;
                    labels[index] = clusterId;
                } else {
                    labels[index] = -1;
                }
            } else {
                labels[index] = clusterId;
            }
        }
        return labels;
    }

    private expandClusterOrder(point: number, processed: boolean[], distanceMatrix: number[][]): void {
        const neighbors = this.regionQuery(point, distanceMatrix);
        processed[point] = true;
        this.ordering.push(point);
        if (this.coreDistances[point] === Infinity) return;
        const seeds: number[] = [];
        this.update(point, neighbors, seeds, processed, distanceMatrix);
        while (seeds.length) {
            seeds.sort((a, b) => this.reachability[a] - this.reachability[b]);
            const current = seeds.shift()!;
            const currentNeighbors = this.regionQuery(current, distanceMatrix);
            processed[current] = true;
            this.ordering.push(current);
            if (this.coreDistances[current] !== Infinity) {
                this.update(current, currentNeighbors, seeds, processed, distanceMatrix);
            }
        }
    }

    private update(point: number, neighbors: number[], seeds: number[], processed: boolean[], distanceMatrix: number[][]): void {
        for (const nb of neighbors) {
            if (processed[nb]) continue;
            const newReach = Math.max(this.coreDistances[point], distanceMatrix[point][nb]);
            if (this.reachability[nb] === Infinity) {
                this.reachability[nb] = newReach;
                seeds.push(nb);
            } else if (newReach < this.reachability[nb]) {
                this.reachability[nb] = newReach;
            }
        }
    }

    private regionQuery(point: number, distanceMatrix: number[][]): number[] {
        const neighbors: number[] = [];
        for (let i = 0; i < distanceMatrix.length; i++) {
            if (distanceMatrix[point][i] <= this.maxEps) {
                neighbors.push(i);
            }
        }
        return neighbors;
    }
}

