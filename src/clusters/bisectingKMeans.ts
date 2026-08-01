import { ClusterBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { validateMatrix } from '../utils/numerics';
import { createRandomGenerator } from '../utils/random';
import { KMeans } from './kmeans';

export interface BisectingKMeansProps { nClusters?: number; maxIter?: number; tol?: number; nInit?: number; randomState?: number; bisectingStrategy?: 'biggestInertia' | 'largestCluster'; }
interface Partition { indices: number[]; center: number[]; inertia: number; left?: Partition; right?: Partition; }

export class BisectingKMeans extends ClusterBase {
    private nClusters: number; private maxIter: number; private tol: number; private nInit: number; private randomState?: number; private strategy: 'biggestInertia' | 'largestCluster';
    private centersState: number[][] = []; private labelsState: number[] = []; private inertiaState = 0; private nFeaturesState = 0;
    constructor(props: BisectingKMeansProps = {}) { super(); const { nClusters = 8, maxIter = 300, tol = 1e-4, nInit = 1, randomState, bisectingStrategy = 'biggestInertia' } = props; if (!Number.isInteger(nClusters) || nClusters < 1 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol < 0 || !Number.isInteger(nInit) || nInit < 1 || !['biggestInertia', 'largestCluster'].includes(bisectingStrategy)) throw new Error('invalid BisectingKMeans parameters'); this.nClusters = nClusters; this.maxIter = maxIter; this.tol = tol; this.nInit = nInit; this.randomState = randomState; this.strategy = bisectingStrategy; }
    public getParams(): Params { return { nClusters: this.nClusters, maxIter: this.maxIter, tol: this.tol, nInit: this.nInit, randomState: this.randomState, bisectingStrategy: this.strategy }; }
    private summarize(X: number[][], indices: number[], fallbackCenter?: number[]): Partition { const center = indices.length === 0 ? (fallbackCenter?.slice() ?? new Array(this.nFeaturesState).fill(0)) : Array.from({ length: this.nFeaturesState }, (_, j) => indices.reduce((sum, i) => sum + X[i][j], 0) / indices.length); const inertia = indices.reduce((sum, i) => sum + X[i].reduce((d, value, j) => d + (value - center[j]) ** 2, 0), 0); return { indices, center, inertia }; }
    public fit(X: number[][]): void {
        this.nFeaturesState = validateMatrix(X); if (this.nClusters > X.length) throw new Error('nClusters cannot exceed number of samples');
        const root = this.summarize(X, X.map((_, i) => i)), partitions: Partition[] = [root], random = createRandomGenerator(this.randomState);
        while (partitions.length < this.nClusters) {
            let selected = 0; for (let i = 1; i < partitions.length; i++) if ((this.strategy === 'largestCluster' ? partitions[i].indices.length : partitions[i].inertia) > (this.strategy === 'largestCluster' ? partitions[selected].indices.length : partitions[selected].inertia)) selected = i;
            const parent = partitions.splice(selected, 1)[0], subset = parent.indices.map(i => X[i]);
            const model = new KMeans({ n_clusters: 2, max_iter: this.maxIter, tol: this.tol, n_init: this.nInit, random_state: Math.floor(random() * 2147483646) + 1 }); const local = model.fitPredict(subset);
            const localCenters = model.getCentroids() ?? [parent.center, parent.center];
            parent.left = this.summarize(X, parent.indices.filter((_, i) => local[i] === 0), localCenters[0]); parent.right = this.summarize(X, parent.indices.filter((_, i) => local[i] === 1), localCenters[1]);
            const compareCenters = (a: number[], b: number[]) => { for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) return a[j] - b[j]; return 0; };
            if (compareCenters(parent.left.center, parent.right.center) > 0) [parent.left, parent.right] = [parent.right, parent.left];
            partitions.push(parent.left, parent.right);
        }
        const leaves: Partition[] = []; const visit = (node: Partition) => { if (node.left && node.right) { visit(node.left); visit(node.right); } else leaves.push(node); }; visit(root);
        this.centersState = leaves.map(partition => partition.center); this.labelsState = new Array(X.length); leaves.forEach((partition, label) => partition.indices.forEach(i => this.labelsState[i] = label)); this.inertiaState = leaves.reduce((sum, partition) => sum + partition.inertia, 0);
    }
    public predict(X: number[][]): number[] { if (this.centersState.length === 0) throw new Error('BisectingKMeans is not fitted'); return X.map(row => { let best = 0, distance = Infinity; this.centersState.forEach((center, i) => { const d = row.reduce((sum, value, j) => sum + (value - center[j]) ** 2, 0); if (d < distance) { distance = d; best = i; } }); return best; }); }
    public fitPredict(X: number[][]): number[] { this.fit(X); return this.labelsState.slice(); }
    public get clusterCenters(): number[][] { return this.centersState.map(row => row.slice()); }
    public get inertia(): number { return this.inertiaState; }
}
registerEstimator('BisectingKMeans', BisectingKMeans);
