import { ClusterBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { validateMatrix } from '../utils/numerics';
import { AgglomerativeClustering } from './agglomerativeClustering';

interface CFSubcluster { count: number; sum: number[]; squaredSum: number; child?: CFNode; }
interface CFNode { leaf: boolean; subclusters: CFSubcluster[]; }
export interface BirchProps { threshold?: number; branchingFactor?: number; nClusters?: number | null; }

function squaredDistance(a: number[], b: number[]): number { return a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0); }

/** Balanced CF-tree with threshold absorption and optional agglomerative global clustering. */
export class Birch extends ClusterBase {
    private threshold: number; private branchingFactor: number; private nClusters: number | null;
    private rootState?: CFNode; private subclusterCentersState: number[][] = []; private subclusterLabelsState: number[] = []; private nFeaturesState = 0;
    constructor(props: BirchProps = {}) {
        super(); const { threshold = .5, branchingFactor = 50, nClusters = 3 } = props;
        if (!Number.isFinite(threshold) || threshold <= 0 || !Number.isInteger(branchingFactor) || branchingFactor < 2 || nClusters !== null && (!Number.isInteger(nClusters) || nClusters < 1)) throw new Error('invalid Birch parameters');
        this.threshold = threshold; this.branchingFactor = branchingFactor; this.nClusters = nClusters;
    }
    public getParams(): Params { return { threshold: this.threshold, branchingFactor: this.branchingFactor, nClusters: this.nClusters }; }
    private center(cluster: CFSubcluster): number[] { return cluster.sum.map(value => value / cluster.count); }
    private singleton(row: number[]): CFSubcluster { return { count: 1, sum: row.slice(), squaredSum: row.reduce((sum, value) => sum + value * value, 0) }; }
    private aggregate(node: CFNode): CFSubcluster {
        const first = node.subclusters[0], sum = new Array(first.sum.length).fill(0); let count = 0, squaredSum = 0;
        for (const cluster of node.subclusters) { count += cluster.count; squaredSum += cluster.squaredSum; cluster.sum.forEach((value, j) => sum[j] += value); }
        return { count, sum, squaredSum, child: node };
    }
    private nearest(subclusters: CFSubcluster[], center: number[]): number {
        let best = 0, distance = Infinity;
        for (let i = 0; i < subclusters.length; i++) { const candidate = squaredDistance(center, this.center(subclusters[i])); if (candidate < distance) { distance = candidate; best = i; } }
        return best;
    }
    private tryAbsorb(cluster: CFSubcluster, row: number[]): boolean {
        const count = cluster.count + 1, sum = cluster.sum.map((value, j) => value + row[j]), squaredSum = cluster.squaredSum + row.reduce((total, value) => total + value * value, 0);
        const centerNorm = sum.reduce((total, value) => total + (value / count) ** 2, 0), radiusSquared = Math.max(0, squaredSum / count - centerNorm);
        if (radiusSquared > this.threshold ** 2 + 1e-14) return false;
        cluster.count = count; cluster.sum = sum; cluster.squaredSum = squaredSum; return true;
    }
    /** Split an overflowing node around its farthest pair; mutate node to the left half and return the right sibling. */
    private split(node: CFNode): CFNode {
        let seedA = 0, seedB = 1, farthest = -Infinity;
        for (let i = 0; i < node.subclusters.length; i++) for (let j = i + 1; j < node.subclusters.length; j++) {
            const distance = squaredDistance(this.center(node.subclusters[i]), this.center(node.subclusters[j]));
            if (distance > farthest) { farthest = distance; seedA = i; seedB = j; }
        }
        const centerA = this.center(node.subclusters[seedA]), centerB = this.center(node.subclusters[seedB]), left: CFSubcluster[] = [], right: CFSubcluster[] = [];
        node.subclusters.forEach((cluster, index) => {
            if (index === seedA) left.push(cluster);
            else if (index === seedB) right.push(cluster);
            else if (squaredDistance(this.center(cluster), centerA) <= squaredDistance(this.center(cluster), centerB)) left.push(cluster);
            else right.push(cluster);
        });
        node.subclusters = left; return { leaf: node.leaf, subclusters: right };
    }
    /** Insert one sample, returning a new sibling when this node overflows. */
    private insert(node: CFNode, row: number[]): CFNode | null {
        if (node.leaf) {
            if (node.subclusters.length === 0 || !this.tryAbsorb(node.subclusters[this.nearest(node.subclusters, row)], row)) node.subclusters.push(this.singleton(row));
        } else {
            const index = this.nearest(node.subclusters, row), entry = node.subclusters[index], child = entry.child!;
            const sibling = this.insert(child, row); node.subclusters[index] = this.aggregate(child);
            if (sibling) node.subclusters.push(this.aggregate(sibling));
        }
        return node.subclusters.length > this.branchingFactor ? this.split(node) : null;
    }
    private leafSubclusters(node: CFNode, output: CFSubcluster[]): void {
        if (node.leaf) { output.push(...node.subclusters); return; }
        for (const cluster of node.subclusters) this.leafSubclusters(cluster.child!, output);
    }
    private finalize(): void {
        const leaves: CFSubcluster[] = []; if (this.rootState) this.leafSubclusters(this.rootState, leaves);
        this.subclusterCentersState = leaves.map(cluster => this.center(cluster));
        if (this.nClusters === null || this.nClusters >= leaves.length) this.subclusterLabelsState = leaves.map((_, i) => i);
        else this.subclusterLabelsState = new AgglomerativeClustering({ nClusters: this.nClusters, linkage: 'ward' }).fitPredict(this.subclusterCentersState);
    }
    public partialFit(X: number[][]): this {
        const features = validateMatrix(X); if (this.nFeaturesState !== 0 && features !== this.nFeaturesState) throw new Error('feature count differs from previous partialFit'); this.nFeaturesState = features;
        if (!this.rootState) this.rootState = { leaf: true, subclusters: [] };
        for (const row of X) { const sibling = this.insert(this.rootState, row); if (sibling) this.rootState = { leaf: false, subclusters: [this.aggregate(this.rootState), this.aggregate(sibling)] }; }
        this.finalize(); return this;
    }
    public fit(X: number[][]): void { this.rootState = undefined; this.nFeaturesState = 0; this.subclusterCentersState = []; this.subclusterLabelsState = []; this.partialFit(X); }
    public predict(X: number[][]): number[] {
        if (this.subclusterCentersState.length === 0) throw new Error('Birch is not fitted');
        return X.map(row => { if (row.length !== this.nFeaturesState) throw new Error('feature count differs from fitted Birch'); return this.subclusterLabelsState[this.nearest(this.subclusterCentersState.map(center => ({ count: 1, sum: center, squaredSum: 0 })), row)]; });
    }
    public fitPredict(X: number[][]): number[] { this.fit(X); return this.predict(X); }
    public get subclusterCenters(): number[][] { return this.subclusterCentersState.map(row => row.slice()); }
    public get subclusterLabels(): number[] { return this.subclusterLabelsState.slice(); }
}
registerEstimator('Birch', Birch);
