import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';

export interface DBScanProps {
    /** neighborhood radius */
    eps?: number;
    /** minimum neighborhood size (including the point itself) for a core point */
    minSamples?: number;
    /** distance metric name */
    distanceType?: Distance.IDistanceType;
}

export class DBScan extends ClusterBase {
    private eps: number;
    private minSamples: number;
    private distanceType: Distance.IDistanceType;
    constructor(props?: DBScanProps);
    /** @deprecated positional form; prefer the props-object constructor */
    constructor(eps?: number, minSamples?: number, distanceType?: Distance.IDistanceType);
    constructor(arg0: DBScanProps | number = {}, minSamplesArg: number = 5, distanceTypeArg: Distance.IDistanceType = 'euclidean') {
        super();
        const props: DBScanProps = typeof arg0 === 'number'
            ? { eps: arg0, minSamples: minSamplesArg, distanceType: distanceTypeArg }
            : arg0;
        const { eps = 0.5, minSamples = 5, distanceType = 'euclidean' } = props;
        this.eps = eps;
        this.minSamples = minSamples;
        this.distanceType = distanceType;
        Distance.useDistance(distanceType); // validate the metric name eagerly
    }

    public getParams(): Params {
        return {
            eps: this.eps,
            minSamples: this.minSamples,
            distanceType: this.distanceType,
        };
    }

    /** resolved lazily from the metric name so instances stay JSON-serializable */
    private get distance(): Distance.IDistance {
        return Distance.useDistance(this.distanceType);
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
        const distance = this.distance;
        for (let i = 0; i < samplesX.length; i++) {
            if (distance(samplesX[index], samplesX[i]) <= this.eps) {
                neighbors.push(i);
            }
        }
        return neighbors;
    }
}
registerEstimator('DBScan', DBScan);
