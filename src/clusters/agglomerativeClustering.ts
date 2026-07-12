import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { Distance } from '../metrics';

/**
 * AgglomerativeClustering: bottom-up hierarchical clustering, following
 * sklearn.cluster.AgglomerativeClustering.
 *
 * - 'ward', 'complete' and 'average' linkage use the nearest-neighbor chain
 *   algorithm (Müllner 2011, the same algorithm scipy's `linkage` and
 *   sklearn use) with Lance-Williams distance updates. NN-chain finds
 *   reciprocal nearest neighbors by walking a chain of nearest neighbors;
 *   every merge is provably identical to the naive O(n^3) greedy
 *   agglomeration for reducible linkages, but runs in O(n^2) time.
 * - 'single' linkage is computed from the minimum spanning tree (Prim on the
 *   dense distance matrix, then Kruskal-style processing of the sorted
 *   edges), exactly like scipy's `mst_single_linkage`.
 *
 * The merge tree is exposed scipy/sklearn-style: `getChildren()` has shape
 * [n-1][2] where values < n are leaves and value n+i refers to the cluster
 * created at merge row i; `getDistances()` gives the merge distances
 * (non-decreasing, rows are sorted by distance like scipy's linkage output).
 *
 * Flat clusters are obtained either by requesting `nClusters` or by cutting
 * all merges with distance >= `distanceThreshold` (exactly one of the two
 * must be set, as in sklearn). Cluster labels are numbered by first sample
 * occurrence (sklearn numbers them in a different, heap-dependent order; the
 * partition itself is identical).
 */

export type AgglomerativeLinkage = 'ward' | 'complete' | 'average' | 'single';

export interface AgglomerativeClusteringProps {
    /**
     * number of flat clusters to extract; must be set to null when
     * `distanceThreshold` is used
     */
    nClusters?: number | null;
    /** linkage criterion */
    linkage?: AgglomerativeLinkage;
    /** distance metric name ('ward' requires 'euclidean') */
    metric?: Distance.IDistanceType;
    /**
     * when set (with nClusters null), merges with distance >= this
     * threshold are not performed and the remaining components become the
     * flat clusters
     */
    distanceThreshold?: number | null;
}

interface LinkageRow {
    left: number;
    right: number;
    distance: number;
    size: number;
}

/**
 * Union-find that assigns a fresh cluster label (starting at n) on every
 * union — the labelling convention of scipy's linkage matrix.
 */
class AggLinkageUnionFind {
    private parent: number[];
    public size: number[];
    private nextLabel: number;
    constructor(n: number) {
        this.parent = new Array(2 * n - 1).fill(-1);
        this.size = new Array(2 * n - 1).fill(0);
        for (let i = 0; i < n; i++) this.size[i] = 1;
        this.nextLabel = n;
    }
    public find(x: number): number {
        let root = x;
        while (this.parent[root] !== -1) root = this.parent[root];
        while (this.parent[x] !== -1) {
            const next = this.parent[x];
            this.parent[x] = root;
            x = next;
        }
        return root;
    }
    public union(a: number, b: number): number {
        this.parent[a] = this.nextLabel;
        this.parent[b] = this.nextLabel;
        this.size[this.nextLabel] = this.size[a] + this.size[b];
        return this.nextLabel++;
    }
}

export class AgglomerativeClustering extends ClusterBase {
    private nClusters: number | null;
    private linkage: AgglomerativeLinkage;
    private metric: Distance.IDistanceType;
    private distanceThreshold: number | null;
    private labels: number[] = [];
    /** merge tree, shape [n-1][2] (sklearn's children_) */
    private children: number[][] = [];
    /** merge distance per row of `children` (sklearn's distances_) */
    private distances: number[] = [];
    /** number of flat clusters found by the last fit (sklearn's n_clusters_) */
    private nClustersFitted: number = 0;

    constructor(props: AgglomerativeClusteringProps = {}) {
        super();
        const {
            nClusters = 2,
            linkage = 'ward',
            metric = 'euclidean',
            distanceThreshold = null,
        } = props;
        if (linkage !== 'ward' && linkage !== 'complete' && linkage !== 'average' && linkage !== 'single') {
            throw new Error(`Unknown linkage "${linkage}"; expected 'ward', 'complete', 'average' or 'single'`);
        }
        if (linkage === 'ward' && metric !== 'euclidean' && metric !== '2-norm') {
            throw new Error(`"${metric}" was provided as metric. Ward can only work with euclidean distances.`);
        }
        if (distanceThreshold !== null && nClusters !== null) {
            throw new Error('Exactly one of nClusters and distanceThreshold has to be set, and the other needs to be null.');
        }
        if (distanceThreshold === null && nClusters === null) {
            throw new Error('Exactly one of nClusters and distanceThreshold has to be set, and the other needs to be null.');
        }
        Distance.useDistance(metric); // validate the metric name eagerly
        this.nClusters = nClusters;
        this.linkage = linkage;
        this.metric = metric;
        this.distanceThreshold = distanceThreshold;
    }

    public getParams(): Params {
        return {
            nClusters: this.nClusters,
            linkage: this.linkage,
            metric: this.metric,
            distanceThreshold: this.distanceThreshold,
        };
    }

    /** resolved lazily from the metric name so instances stay JSON-serializable */
    private get distance(): Distance.IDistance {
        return Distance.useDistance(this.metric);
    }

    public fitPredict(samplesX: number[][]): number[] {
        const n = samplesX.length;
        this.labels = [];
        this.children = [];
        this.distances = [];
        this.nClustersFitted = 0;
        if (n === 0) return [];
        if (this.nClusters !== null && (this.nClusters < 1 || this.nClusters > n)) {
            throw new Error(`Cannot extract ${this.nClusters} clusters from ${n} samples`);
        }
        if (n === 1) {
            this.labels = [0];
            this.nClustersFitted = 1;
            return this.labels;
        }

        const dist = this.pairwiseDistances(samplesX);
        const merges = this.linkage === 'single'
            ? AgglomerativeClustering.mstSingleLinkage(dist)
            : this.nnChain(dist);

        // sort merges by distance (stable, like np.argsort(..., kind='stable')
        // in scipy) and relabel through the union-find so internal nodes get
        // the scipy convention: merge row i creates cluster n + i
        merges.sort((a, b) => a.distance - b.distance);
        const uf = new AggLinkageUnionFind(n);
        const rows: LinkageRow[] = [];
        for (const m of merges) {
            const rootA = uf.find(m.left);
            const rootB = uf.find(m.right);
            const left = Math.min(rootA, rootB);
            const right = Math.max(rootA, rootB);
            const size = uf.size[left] + uf.size[right];
            uf.union(rootA, rootB);
            rows.push({ left, right, distance: m.distance, size });
        }
        this.children = rows.map((r) => [r.left, r.right]);
        this.distances = rows.map((r) => r.distance);

        const nClusters = this.distanceThreshold !== null
            // sklearn: n_clusters_ = count(distances_ >= distance_threshold) + 1
            ? this.distances.filter((d) => d >= (this.distanceThreshold as number)).length + 1
            : (this.nClusters as number);
        this.nClustersFitted = nClusters;
        this.labels = this.cutTree(n, nClusters);
        return this.labels;
    }

    public getLabels(): number[] {
        return this.labels;
    }

    /** merge tree from the last fit, shape [n-1][2] (sklearn's children_) */
    public getChildren(): number[][] {
        return this.children;
    }

    /** merge distances from the last fit, one per row of getChildren() */
    public getDistances(): number[] {
        return this.distances;
    }

    /** number of flat clusters found by the last fit */
    public getNClusters(): number {
        return this.nClustersFitted;
    }

    private pairwiseDistances(samplesX: number[][]): number[][] {
        const n = samplesX.length;
        const distance = this.distance;
        const dist: number[][] = new Array(n);
        for (let i = 0; i < n; i++) dist[i] = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const d = distance(samplesX[i], samplesX[j]);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        return dist;
    }

    /** Lance-Williams distance update for the merged cluster (x ∪ y) vs i. */
    private newDistance(dxi: number, dyi: number, dxy: number, nx: number, ny: number, ni: number): number {
        switch (this.linkage) {
            case 'complete':
                return Math.max(dxi, dyi);
            case 'average':
                return (nx * dxi + ny * dyi) / (nx + ny);
            case 'ward':
                // recurrence on plain euclidean distances (scipy's `_ward`)
                return Math.sqrt(
                    ((ni + nx) * dxi * dxi + (ni + ny) * dyi * dyi - ni * dxy * dxy) / (nx + ny + ni)
                );
            default:
                throw new Error(`No Lance-Williams update for linkage "${this.linkage}"`);
        }
    }

    /**
     * Nearest-neighbor chain agglomeration (scipy's `nn_chain`). `dist` is
     * mutated in place: row/column y holds the distances of the merged
     * cluster after each merge; dead clusters keep size 0.
     *
     * Returned merges reference *leaf representative* indices (as in scipy):
     * the sorted-relabel step afterwards converts them into cluster labels.
     */
    private nnChain(dist: number[][]): LinkageRow[] {
        const n = dist.length;
        const size: number[] = new Array(n).fill(1);
        const chain: number[] = new Array(n).fill(0);
        let chainLength = 0;
        const merges: LinkageRow[] = [];
        for (let k = 0; k < n - 1; k++) {
            if (chainLength === 0) {
                chainLength = 1;
                for (let i = 0; i < n; i++) {
                    if (size[i] > 0) {
                        chain[0] = i;
                        break;
                    }
                }
            }
            let x = 0;
            let y = -1;
            let currentMin = Infinity;
            // extend the chain until two clusters are mutual nearest neighbors
            for (;;) {
                x = chain[chainLength - 1];
                if (chainLength > 1) {
                    y = chain[chainLength - 2];
                    currentMin = dist[x][y];
                } else {
                    y = -1;
                    currentMin = Infinity;
                }
                for (let i = 0; i < n; i++) {
                    if (size[i] === 0 || i === x) continue;
                    const d = dist[x][i];
                    if (d < currentMin) {
                        currentMin = d;
                        y = i;
                    }
                }
                if (chainLength > 1 && y === chain[chainLength - 2]) break;
                chain[chainLength] = y;
                chainLength += 1;
            }
            chainLength -= 2;
            if (x > y) {
                const tmp = x;
                x = y;
                y = tmp;
            }
            const nx = size[x];
            const ny = size[y];
            merges.push({ left: x, right: y, distance: currentMin, size: nx + ny });
            size[x] = 0;
            size[y] = nx + ny;
            for (let i = 0; i < n; i++) {
                const ni = size[i];
                if (ni === 0 || i === y) continue;
                const d = this.newDistance(dist[x][i], dist[y][i], currentMin, nx, ny, ni);
                dist[y][i] = d;
                dist[i][y] = d;
            }
        }
        return merges;
    }

    /**
     * Single linkage from the minimum spanning tree (scipy's
     * `mst_single_linkage`): Prim's algorithm on the dense distance matrix.
     * The MST edges are exactly the single-linkage merges once sorted.
     */
    private static mstSingleLinkage(dist: number[][]): LinkageRow[] {
        const n = dist.length;
        const merged: boolean[] = new Array(n).fill(false);
        const minDist: number[] = new Array(n).fill(Infinity);
        const nearestSource: number[] = new Array(n).fill(0);
        const merges: LinkageRow[] = [];
        let x = 0;
        for (let k = 0; k < n - 1; k++) {
            merged[x] = true;
            let y = -1;
            let currentMin = Infinity;
            for (let i = 0; i < n; i++) {
                if (merged[i]) continue;
                const d = dist[x][i];
                if (d < minDist[i]) {
                    minDist[i] = d;
                    nearestSource[i] = x;
                }
                if (minDist[i] < currentMin) {
                    currentMin = minDist[i];
                    y = i;
                }
            }
            merges.push({ left: nearestSource[y], right: y, distance: currentMin, size: 0 });
            x = y;
        }
        return merges;
    }

    /**
     * Flat clusters from the first n - nClusters merge rows (rows are sorted
     * by distance, so stopping early is exactly sklearn's `_hc_cut`, which
     * repeatedly re-splits the most recent merge). Labels are numbered by
     * first sample occurrence.
     */
    private cutTree(n: number, nClusters: number): number[] {
        const used = n - nClusters;
        const parent: number[] = new Array(n + used).fill(-1);
        for (let i = 0; i < used; i++) {
            parent[this.children[i][0]] = n + i;
            parent[this.children[i][1]] = n + i;
        }
        const find = (node: number): number => {
            let root = node;
            while (parent[root] !== -1) root = parent[root];
            while (parent[node] !== -1) {
                const next = parent[node];
                parent[node] = root;
                node = next;
            }
            return root;
        };
        const labels: number[] = new Array(n).fill(-1);
        const rootLabel: Map<number, number> = new Map();
        for (let i = 0; i < n; i++) {
            const root = find(i);
            let label = rootLabel.get(root);
            if (label === undefined) {
                label = rootLabel.size;
                rootLabel.set(root, label);
            }
            labels[i] = label;
        }
        return labels;
    }
}
registerEstimator('AgglomerativeClustering', AgglomerativeClustering);
