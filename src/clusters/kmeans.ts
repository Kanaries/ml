import { ClusterBase } from '../base/cluster';
import { kmeansPlusPlus } from './kmeans_plusplus';
import { createRandomGenerator } from '../utils/random';

interface AssignmentResult {
    labels: number[];
    inertia: number;
    distances: number[];
}

interface SingleRunResult {
    centers: number[][];
    labels: number[];
    inertia: number;
}

export class KMeans extends ClusterBase {
    private n_clusters: number;
    private tol: number;
    private max_iter: number;
    private n_init: number;
    private random_state?: number;
    private userCenters: number[][] | null;
    private centers: number[][] | null;
    private samplesY: number[];
    private inertia: number;
    /**
     * @param n_clusters number of clusters
     * @param tol relative tolerance on the weighted inertia change used as
     *            convergence criterion (previously named `opt_ratio`; old
     *            positional calls keep working and map to `tol`)
     * @param initCenters optional user-provided initial centers; when given,
     *                    they are used as-is and `n_init` is forced to 1
     * @param max_iter maximum Lloyd iterations per run
     * @param random_state optional seed for reproducible k-means++ init
     * @param n_init number of k-means++ restarts; the run with the lowest
     *               weighted inertia wins
     */
    constructor(n_clusters: number = 2, tol: number = 1e-4, initCenters?: number[][], max_iter: number = 30, random_state?: number, n_init: number = 10) {
        super();
        this.n_clusters = n_clusters;
        this.tol = tol;
        this.max_iter = max_iter;
        this.n_init = n_init;
        this.random_state = random_state;
        this.userCenters = initCenters ? initCenters.map((center) => center.slice()) : null;
        this.centers = null;
        this.samplesY = [];
        this.inertia = Infinity;
    }
    private assignLabels(samplesX: number[][], centers: number[][], sampleWeights: number[]): AssignmentResult {
        const labels: number[] = new Array(samplesX.length).fill(0);
        const distances: number[] = new Array(samplesX.length).fill(0);
        let inertia = 0;
        for (let i = 0; i < samplesX.length; i++) {
            let nearestIndex = 0;
            let nearestDis = Infinity;
            for (let j = 0; j < centers.length; j++) {
                let disSquare = 0;
                for (let d = 0; d < centers[j].length; d++) {
                    const diff = centers[j][d] - samplesX[i][d];
                    disSquare += diff * diff;
                }
                if (disSquare < nearestDis) {
                    nearestDis = disSquare;
                    nearestIndex = j;
                }
            }
            labels[i] = nearestIndex;
            distances[i] = nearestDis;
            inertia += nearestDis * sampleWeights[i];
        }
        return { labels, inertia, distances };
    }
    private updateCentroids(samplesX: number[][], labels: number[], sampleWeights: number[], distances: number[], prevCenters: number[][]): number[][] {
        const { n_clusters } = this;
        const featureSize = samplesX[0].length;
        const centers: number[][] = [];
        for (let i = 0; i < n_clusters; i++) {
            centers.push(new Array(featureSize).fill(0));
        }
        const neighborCounter: number[] = new Array(n_clusters).fill(0);
        for (let i = 0; i < samplesX.length; i++) {
            const centerIndex = labels[i];
            neighborCounter[centerIndex] += sampleWeights[i];
            for (let j = 0; j < featureSize; j++) {
                centers[centerIndex][j] += samplesX[i][j] * sampleWeights[i];
            }
        }
        // squared distance of each sample to its nearest center; consumed
        // when relocating empty clusters so a sample is only used once
        const relocationDist = distances.slice();
        for (let i = 0; i < n_clusters; i++) {
            if (neighborCounter[i] > 0) {
                for (let j = 0; j < featureSize; j++) {
                    centers[i][j] /= neighborCounter[i];
                }
            } else {
                // empty cluster: relocate its center to the sample farthest
                // from its nearest center (sklearn-style), never produce NaN
                let farthestIndex = -1;
                let farthestDist = -Infinity;
                for (let s = 0; s < samplesX.length; s++) {
                    if (relocationDist[s] > farthestDist) {
                        farthestDist = relocationDist[s];
                        farthestIndex = s;
                    }
                }
                if (farthestIndex !== -1) {
                    centers[i] = samplesX[farthestIndex].slice();
                    relocationDist[farthestIndex] = -Infinity;
                } else {
                    centers[i] = prevCenters[i].slice();
                }
            }
        }
        return centers;
    }
    private runSingle(samplesX: number[][], sampleWeights: number[], initialCenters: number[][]): SingleRunResult {
        let centers = initialCenters.map((center) => center.slice());
        let labels: number[] = [];
        let inertia = Infinity;
        let prevInertia = Infinity;
        for (let iter = 0; iter < this.max_iter; iter++) {
            const assignment = this.assignLabels(samplesX, centers, sampleWeights);
            labels = assignment.labels;
            inertia = assignment.inertia;
            if (prevInertia !== Infinity) {
                if (prevInertia === 0 || Math.abs(prevInertia - inertia) / prevInertia < this.tol) {
                    break;
                }
            }
            prevInertia = inertia;
            centers = this.updateCentroids(samplesX, labels, sampleWeights, assignment.distances, centers);
        }
        return { centers, labels, inertia };
    }
    public fitPredict(sampleX: number[][], sampleWeights?: number[]): number[] {
        // fitting always starts from a clean state: previous centers are
        // discarded (user-provided initCenters are reused each fit)
        this.centers = null;
        this.samplesY = [];
        this.inertia = Infinity;
        if (sampleX.length === 0) {
            return [];
        }
        const weights = sampleWeights ? sampleWeights : sampleX.map(() => 1);
        const rng = createRandomGenerator(this.random_state);
        const runs = this.userCenters ? 1 : Math.max(1, this.n_init);
        let best: SingleRunResult | null = null;
        for (let run = 0; run < runs; run++) {
            const initialCenters = this.userCenters
                ? this.userCenters.map((center) => center.slice())
                : kmeansPlusPlus(sampleX, this.n_clusters, weights, rng).centers;
            const result = this.runSingle(sampleX, weights, initialCenters);
            if (best === null || result.inertia < best.inertia) {
                best = result;
            }
        }
        this.centers = best.centers;
        this.samplesY = best.labels;
        this.inertia = best.inertia;
        return this.samplesY;
    }
    public getCentroids(): number[][] | null {
        return this.centers;
    }
    public getInertia(): number {
        return this.inertia;
    }
}
