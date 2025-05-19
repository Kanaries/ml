import { ClusterBase } from '../base/cluster';
import { Distance } from '../metrics';

export class DBScan extends ClusterBase {
    private eps: number;
    private minSamples: number;
    private distance: Distance.IDistance;
    constructor(eps: number = 0.5, minSamples: number = 5, distanceType: Distance.IDistanceType = 'euclidiean') {
        super();
        this.eps = eps;
        this.minSamples = minSamples;
        this.distance = Distance.useDistance(distanceType);
    }

    public fitPredict(samplesX: number[][]): number[] {
        const labels: Array<number | undefined> = new Array(samplesX.length);
        const visited: boolean[] = new Array(samplesX.length).fill(false);
        let clusterId = 0;
        for (let i = 0; i < samplesX.length; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            const neighbors = this.regionQuery(samplesX, i);
            if (neighbors.length < this.minSamples) {
                labels[i] = -1;
            } else {
                this.expandCluster(samplesX, i, neighbors, clusterId, labels, visited);
                clusterId++;
            }
        }
        return labels as number[];
    }

    private expandCluster(samplesX: number[][], pointIndex: number, neighbors: number[], clusterId: number, labels: Array<number | undefined>, visited: boolean[]) {
        labels[pointIndex] = clusterId;
        for (let i = 0; i < neighbors.length; i++) {
            const nIndex = neighbors[i];
            if (!visited[nIndex]) {
                visited[nIndex] = true;
                const nNeighbors = this.regionQuery(samplesX, nIndex);
                if (nNeighbors.length >= this.minSamples) {
                    for (const nb of nNeighbors) {
                        if (neighbors.indexOf(nb) === -1) {
                            neighbors.push(nb);
                        }
                    }
                }
            }
            if (labels[nIndex] === undefined || labels[nIndex] === -1) {
                labels[nIndex] = clusterId;
            }
        }
    }

    private regionQuery(samplesX: number[][], index: number): number[] {
        const neighbors: number[] = [];
        for (let i = 0; i < samplesX.length; i++) {
            if (this.distance(samplesX[index], samplesX[i]) <= this.eps) {
                neighbors.push(i);
            }
        }
        return neighbors;
    }
}
