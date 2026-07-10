import { ClusterBase } from '../base/cluster';
import { Distance } from '../metrics';

/**
 * HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications
 * with Noise.
 *
 * Full implementation following sklearn.cluster.HDBSCAN (>= 1.3):
 *   1. core distances (distance to the min_samples-th nearest neighbor,
 *      counting the point itself, as sklearn does);
 *   2. mutual reachability distances
 *      d_mreach(a, b) = max(core_a, core_b, d(a, b));
 *   3. minimum spanning tree of the mutual reachability graph (Prim);
 *   4. single linkage hierarchy, condensed by min_cluster_size, and flat
 *      cluster extraction via Excess of Mass (EOM) stability, optionally
 *      post-processed with cluster_selection_epsilon.
 *
 * Points that do not belong to any selected cluster are labelled -1.
 */

interface IHierarchyNode {
    left: number;
    right: number;
    value: number;
    size: number;
}

interface ICondensedEdge {
    parent: number;
    child: number;
    lambda: number;
    size: number;
}

/**
 * Port of numpy's argsort for float64 with the default kind='quicksort'
 * (introsort: median-of-3 quicksort + insertion sort below 16 elements +
 * heapsort fallback). sklearn sorts the MST edges with np.argsort before
 * building the single linkage tree; equal-weight edges (which are common in
 * mutual reachability graphs, where several pairs share the same dominating
 * core distance) end up in an implementation-defined order that can change
 * the condensed tree. Reproducing the exact permutation keeps labels
 * identical to sklearn even on tied edges.
 */
function numpyArgsort(values: number[]): number[] {
    const n = values.length;
    const tosort: number[] = new Array(n);
    for (let i = 0; i < n; i++) tosort[i] = i;
    if (n < 2) return tosort;
    const lt = (a: number, b: number): boolean => a < b || (b !== b && a === a);
    const swap = (i: number, j: number) => {
        const tmp = tosort[i];
        tosort[i] = tosort[j];
        tosort[j] = tmp;
    };
    const heapsort = (offset: number, len: number) => {
        // numpy aheapsort on tosort[offset .. offset + len - 1], 1-based
        const a = (i: number) => tosort[offset + i - 1];
        const setA = (i: number, v: number) => {
            tosort[offset + i - 1] = v;
        };
        let nn = len;
        for (let l = nn >> 1; l > 0; l--) {
            const tmp = a(l);
            let i = l;
            for (let j = l * 2; j <= nn; ) {
                if (j < nn && lt(values[a(j)], values[a(j + 1)])) j += 1;
                if (lt(values[tmp], values[a(j)])) {
                    setA(i, a(j));
                    i = j;
                    j += i;
                } else {
                    break;
                }
            }
            setA(i, tmp);
        }
        for (; nn > 1; ) {
            const tmp = a(nn);
            setA(nn, a(1));
            nn -= 1;
            let i = 1;
            for (let j = 2; j <= nn; ) {
                if (j < nn && lt(values[a(j)], values[a(j + 1)])) j++;
                if (lt(values[tmp], values[a(j)])) {
                    setA(i, a(j));
                    i = j;
                    j += i;
                } else {
                    break;
                }
            }
            setA(i, tmp);
        }
    };

    const SMALL_QUICKSORT = 15;
    const stack: number[] = [];
    let pl = 0;
    let pr = n - 1;
    let depthLimit = 2 * Math.floor(Math.log2(n));
    for (;;) {
        let heapsorted = false;
        while (pr - pl > SMALL_QUICKSORT) {
            if (depthLimit-- < 0) {
                heapsort(pl, pr - pl + 1);
                heapsorted = true;
                break;
            }
            let pm = pl + ((pr - pl) >> 1);
            if (lt(values[tosort[pm]], values[tosort[pl]])) swap(pm, pl);
            if (lt(values[tosort[pr]], values[tosort[pm]])) swap(pr, pm);
            if (lt(values[tosort[pm]], values[tosort[pl]])) swap(pm, pl);
            const vp = values[tosort[pm]];
            let pi = pl;
            let pj = pr - 1;
            swap(pm, pj);
            for (;;) {
                do {
                    pi++;
                } while (lt(values[tosort[pi]], vp));
                do {
                    pj--;
                } while (lt(vp, values[tosort[pj]]));
                if (pi >= pj) break;
                swap(pi, pj);
            }
            const pk = pr - 1;
            swap(pi, pk);
            if (pi - pl < pr - pi) {
                stack.push(pi + 1, pr);
                pr = pi - 1;
            } else {
                stack.push(pl, pi - 1);
                pl = pi + 1;
            }
        }
        if (!heapsorted) {
            for (let pi = pl + 1; pi <= pr; pi++) {
                const vi = tosort[pi];
                const vp = values[vi];
                let pj = pi;
                let pk = pi - 1;
                while (pj > pl && lt(vp, values[tosort[pk]])) {
                    tosort[pj--] = tosort[pk--];
                }
                tosort[pj] = vi;
            }
        }
        if (stack.length === 0) break;
        pr = stack.pop()!;
        pl = stack.pop()!;
    }
    return tosort;
}

/**
 * Union-find used to convert a sorted MST edge list into a single linkage
 * hierarchy. Mirrors sklearn.cluster._hierarchical_fast.UnionFind: every
 * union creates a fresh cluster label starting at n.
 */
class LinkageUnionFind {
    private parent: number[];
    public size: number[];
    private nextLabel: number;
    constructor(n: number) {
        this.parent = new Array(2 * n - 1).fill(-1);
        this.nextLabel = n;
        this.size = new Array(2 * n - 1).fill(0);
        for (let i = 0; i < n; i++) this.size[i] = 1;
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

    public union(m: number, n: number): void {
        this.parent[m] = this.nextLabel;
        this.parent[n] = this.nextLabel;
        this.size[this.nextLabel] = this.size[m] + this.size[n];
        this.nextLabel++;
    }
}

export class HDBScan extends ClusterBase {
    private minClusterSize: number;
    private minSamples: number;
    private epsilon: number;
    private allowSingleCluster: boolean;
    private distance: Distance.IDistance;
    private labels: number[] = [];
    private probabilities: number[] = [];
    constructor(
        min_cluster_size: number = 5,
        min_samples: number | null = null,
        cluster_selection_epsilon: number = 0.0,
        metric: Distance.IDistanceType = 'euclidean',
        allow_single_cluster: boolean = false
    ) {
        super();
        this.minClusterSize = Math.max(2, min_cluster_size);
        this.minSamples = min_samples === null ? min_cluster_size : min_samples;
        this.epsilon = cluster_selection_epsilon;
        this.allowSingleCluster = allow_single_cluster;
        this.distance = Distance.useDistance(metric);
    }

    public fitPredict(samplesX: number[][]): number[] {
        const n = samplesX.length;
        if (n === 0) {
            this.labels = [];
            this.probabilities = [];
            return [];
        }
        if (n === 1) {
            this.labels = [-1];
            this.probabilities = [0];
            return [-1];
        }
        const mutualReachability = this.mutualReachability(samplesX);
        const mst = this.primMST(mutualReachability);
        const order = numpyArgsort(mst.map(e => e[2]));
        const sortedMst = order.map(i => mst[i]);
        const hierarchy = this.singleLinkage(sortedMst, n);
        const condensed = this.condenseTree(hierarchy, n);
        this.extractClusters(condensed, n);
        return this.labels;
    }

    public getLabels(): number[] {
        return this.labels;
    }

    /**
     * Cluster membership strength of each sample from the last fitPredict
     * call. Noise points have probability 0.
     */
    public getProbabilities(): number[] {
        return this.probabilities;
    }

    /**
     * Compute the mutual reachability matrix. The core distance of a point
     * is its distance to the min_samples-th nearest neighbor, where the
     * point itself is counted (sklearn semantics).
     */
    private mutualReachability(samplesX: number[][]): number[][] {
        const n = samplesX.length;
        const dist: number[][] = new Array(n);
        for (let i = 0; i < n; i++) dist[i] = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const d = this.distance(samplesX[i], samplesX[j]);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        // clamp so tiny datasets do not crash instead of raising like sklearn
        const k = Math.min(this.minSamples, n) - 1;
        const core: number[] = new Array(n);
        for (let i = 0; i < n; i++) {
            const row = dist[i].slice().sort((a, b) => a - b);
            core[i] = row[k];
        }
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                dist[i][j] = Math.max(core[i], core[j], dist[i][j]);
            }
        }
        return dist;
    }

    /**
     * Prim's algorithm on the dense mutual reachability graph. Returns the
     * MST as [from, to, weight] edges. Mirrors sklearn's
     * mst_from_mutual_reachability exactly, including first-minimum
     * tie-breaking and recording the most recently added node (rather than
     * the true argmin source) as the from-endpoint, which matters for tie
     * resolution in the subsequent single linkage step.
     */
    private primMST(graph: number[][]): Array<[number, number, number]> {
        const n = graph.length;
        const mst: Array<[number, number, number]> = [];
        const inTree: boolean[] = new Array(n).fill(false);
        const minReach: number[] = new Array(n).fill(Infinity);
        let currentNode = 0;
        for (let i = 0; i < n - 1; i++) {
            inTree[currentNode] = true;
            let newNode = -1;
            let best = Infinity;
            for (let j = 0; j < n; j++) {
                if (inTree[j]) continue;
                const right = graph[currentNode][j];
                if (right < minReach[j]) {
                    minReach[j] = right;
                }
                if (minReach[j] < best) {
                    best = minReach[j];
                    newNode = j;
                }
            }
            mst.push([currentNode, newNode, minReach[newNode]]);
            currentNode = newNode;
        }
        return mst;
    }

    /**
     * Convert the weight-sorted MST edge list into a single linkage
     * hierarchy in scipy format: row i merges two clusters into cluster
     * n + i.
     */
    private singleLinkage(mst: Array<[number, number, number]>, n: number): IHierarchyNode[] {
        const uf = new LinkageUnionFind(n);
        const hierarchy: IHierarchyNode[] = [];
        for (let i = 0; i < mst.length; i++) {
            const rootLeft = uf.find(mst[i][0]);
            const rootRight = uf.find(mst[i][1]);
            hierarchy.push({
                left: rootLeft,
                right: rootRight,
                value: mst[i][2],
                size: uf.size[rootLeft] + uf.size[rootRight]
            });
            uf.union(rootLeft, rootRight);
        }
        return hierarchy;
    }

    private bfsFromHierarchy(hierarchy: IHierarchyNode[], root: number, n: number): number[] {
        const result: number[] = [];
        let queue: number[] = [root];
        while (queue.length > 0) {
            result.push(...queue);
            const next: number[] = [];
            for (const node of queue) {
                if (node >= n) {
                    next.push(hierarchy[node - n].left, hierarchy[node - n].right);
                }
            }
            queue = next;
        }
        return result;
    }

    /**
     * Condense the single linkage tree: splits where a side has fewer than
     * min_cluster_size points are treated as points falling out of the
     * parent cluster rather than new clusters. Lambda values are 1/distance.
     */
    private condenseTree(hierarchy: IHierarchyNode[], n: number): ICondensedEdge[] {
        const root = 2 * hierarchy.length;
        const nodeList = this.bfsFromHierarchy(hierarchy, root, n);
        const relabel: number[] = new Array(root + 1).fill(-1);
        relabel[root] = n;
        let nextLabel = n + 1;
        const ignore: boolean[] = new Array(root + 1).fill(false);
        const result: ICondensedEdge[] = [];

        for (const node of nodeList) {
            if (ignore[node] || node < n) continue;
            const children = hierarchy[node - n];
            const lambda = children.value > 0 ? 1 / children.value : Infinity;
            const leftCount = children.left >= n ? hierarchy[children.left - n].size : 1;
            const rightCount = children.right >= n ? hierarchy[children.right - n].size : 1;

            if (leftCount >= this.minClusterSize && rightCount >= this.minClusterSize) {
                relabel[children.left] = nextLabel++;
                result.push({ parent: relabel[node], child: relabel[children.left], lambda, size: leftCount });
                relabel[children.right] = nextLabel++;
                result.push({ parent: relabel[node], child: relabel[children.right], lambda, size: rightCount });
            } else if (leftCount < this.minClusterSize && rightCount < this.minClusterSize) {
                for (const side of [children.left, children.right]) {
                    for (const subNode of this.bfsFromHierarchy(hierarchy, side, n)) {
                        if (subNode < n) {
                            result.push({ parent: relabel[node], child: subNode, lambda, size: 1 });
                        }
                        ignore[subNode] = true;
                    }
                }
            } else {
                const keep = leftCount < this.minClusterSize ? children.right : children.left;
                const drop = leftCount < this.minClusterSize ? children.left : children.right;
                relabel[keep] = relabel[node];
                for (const subNode of this.bfsFromHierarchy(hierarchy, drop, n)) {
                    if (subNode < n) {
                        result.push({ parent: relabel[node], child: subNode, lambda, size: 1 });
                    }
                    ignore[subNode] = true;
                }
            }
        }
        return result;
    }

    /**
     * Stability S(C) = sum over children rows of (lambda - lambda_birth(C))
     * weighted by size.
     */
    private computeStability(condensed: ICondensedEdge[], n: number): Map<number, number> {
        const births: Map<number, number> = new Map();
        for (const edge of condensed) {
            births.set(edge.child, edge.lambda);
        }
        births.set(n, 0); // root
        const stability: Map<number, number> = new Map();
        for (const edge of condensed) {
            const birth = births.get(edge.parent) || 0;
            stability.set(edge.parent, (stability.get(edge.parent) || 0) + (edge.lambda - birth) * edge.size);
        }
        return stability;
    }

    private bfsFromClusterTree(clusterTree: ICondensedEdge[], root: number): number[] {
        const result: number[] = [];
        let queue: number[] = [root];
        while (queue.length > 0) {
            result.push(...queue);
            const frontier = new Set(queue);
            queue = clusterTree.filter(e => frontier.has(e.parent)).map(e => e.child);
        }
        return result;
    }

    /**
     * Walk from a selected cluster towards the root until the birth
     * distance of the parent exceeds cluster_selection_epsilon
     * (sklearn's traverse_upwards).
     */
    private traverseUpwards(
        clusterTree: ICondensedEdge[],
        birthEps: Map<number, number>,
        root: number,
        leaf: number
    ): number {
        const parentEdge = clusterTree.find(e => e.child === leaf);
        if (parentEdge === undefined) return leaf;
        const parent = parentEdge.parent;
        if (parent === root) {
            return this.allowSingleCluster ? parent : leaf;
        }
        if (birthEps.get(parent)! > this.epsilon) {
            return parent;
        }
        return this.traverseUpwards(clusterTree, birthEps, root, parent);
    }

    /** sklearn's epsilon_search over the EOM/leaf-selected clusters. */
    private epsilonSearch(
        selected: number[],
        clusterTree: ICondensedEdge[],
        root: number
    ): Set<number> {
        // birth eps of a cluster = 1 / lambda at which it split off
        const birthEps: Map<number, number> = new Map();
        for (const edge of clusterTree) {
            birthEps.set(edge.child, edge.lambda > 0 ? 1 / edge.lambda : Infinity);
        }
        const chosen: number[] = [];
        const processed: Set<number> = new Set();
        for (const leaf of selected) {
            const eps = birthEps.get(leaf)!;
            if (eps < this.epsilon) {
                if (!processed.has(leaf)) {
                    const epsilonChild = this.traverseUpwards(clusterTree, birthEps, root, leaf);
                    chosen.push(epsilonChild);
                    for (const subNode of this.bfsFromClusterTree(clusterTree, epsilonChild)) {
                        if (subNode !== epsilonChild) processed.add(subNode);
                    }
                }
            } else {
                chosen.push(leaf);
            }
        }
        return new Set(chosen);
    }

    /**
     * EOM cluster selection, labelling and probabilities, mirroring
     * sklearn's _get_clusters / _do_labelling / get_probabilities.
     */
    private extractClusters(condensed: ICondensedEdge[], n: number): void {
        this.labels = new Array(n).fill(-1);
        this.probabilities = new Array(n).fill(0);
        if (condensed.length === 0) return;

        const root = n;
        const stability = this.computeStability(condensed, n);
        const clusterTree = condensed.filter(e => e.size > 1);
        // ids descend from the deepest clusters up to (but excluding) the root
        let nodeList = Array.from(stability.keys()).sort((a, b) => b - a);
        if (!this.allowSingleCluster) {
            nodeList = nodeList.filter(c => c !== root);
        }

        const isCluster: Map<number, boolean> = new Map();
        for (const node of nodeList) isCluster.set(node, true);

        for (const node of nodeList) {
            let subtreeStability = 0;
            for (const edge of clusterTree) {
                if (edge.parent === node) subtreeStability += stability.get(edge.child)!;
            }
            if (subtreeStability > stability.get(node)!) {
                isCluster.set(node, false);
                stability.set(node, subtreeStability);
            } else {
                for (const subNode of this.bfsFromClusterTree(clusterTree, node)) {
                    if (subNode !== node) isCluster.set(subNode, false);
                }
            }
        }

        if (this.epsilon !== 0 && clusterTree.length > 0) {
            const eomClusters = nodeList.filter(c => isCluster.get(c));
            let selected: Set<number> = new Set();
            if (eomClusters.length === 1 && eomClusters[0] === root) {
                if (this.allowSingleCluster) selected = new Set(eomClusters);
            } else {
                selected = this.epsilonSearch(eomClusters, clusterTree, root);
            }
            for (const c of isCluster.keys()) {
                isCluster.set(c, selected.has(c));
            }
        }

        const clusters: number[] = [];
        for (const [c, is] of isCluster) {
            if (is) clusters.push(c);
        }
        clusters.sort((a, b) => a - b);
        const clusterSet = new Set(clusters);
        const clusterLabelMap: Map<number, number> = new Map();
        clusters.forEach((c, idx) => clusterLabelMap.set(c, idx));

        // labelling: each point belongs to its nearest selected ancestor
        const parentOf: Map<number, number> = new Map();
        const lambdaOf: Map<number, number> = new Map();
        for (const edge of condensed) {
            parentOf.set(edge.child, edge.parent);
            lambdaOf.set(edge.child, edge.lambda);
        }
        for (let point = 0; point < n; point++) {
            let cur = parentOf.get(point);
            while (cur !== undefined && cur !== root && !clusterSet.has(cur)) {
                cur = parentOf.get(cur);
            }
            if (cur === undefined) continue;
            if (clusterSet.has(cur) && cur !== root) {
                this.labels[point] = clusterLabelMap.get(cur)!;
            } else if (cur === root && clusterSet.has(root) && clusters.length === 1 && this.allowSingleCluster) {
                // single-cluster special case: only sufficiently dense
                // points join the root cluster
                let threshold: number;
                if (this.epsilon !== 0) {
                    threshold = 1 / this.epsilon;
                } else {
                    threshold = -Infinity;
                    for (const edge of condensed) {
                        if (edge.parent === root) threshold = Math.max(threshold, edge.lambda);
                    }
                }
                if (lambdaOf.get(point)! >= threshold) {
                    this.labels[point] = clusterLabelMap.get(root)!;
                }
            }
        }

        // probabilities: lambda of the point relative to the max lambda of
        // its cluster
        const deaths: Map<number, number> = new Map();
        for (const edge of condensed) {
            deaths.set(edge.parent, Math.max(deaths.get(edge.parent) || 0, edge.lambda));
        }
        for (const edge of condensed) {
            const point = edge.child;
            if (point >= root) continue;
            const label = this.labels[point];
            if (label === -1) continue;
            const cluster = clusters[label];
            const maxLambda = deaths.get(cluster)!;
            if (maxLambda === 0 || !isFinite(edge.lambda)) {
                this.probabilities[point] = 1;
            } else {
                this.probabilities[point] = Math.min(edge.lambda, maxLambda) / maxLambda;
            }
        }
    }
}
