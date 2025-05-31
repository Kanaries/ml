import { Distance } from '../metrics';

interface BallNode {
    centroid: number[];
    radius: number;
    indices: number[];
    left: BallNode | null;
    right: BallNode | null;
}

export class BallTree {
    private X: number[][];
    private leafSize: number;
    private root: BallNode | null;
    private distance: Distance.IDistance;
    private p: number;

    constructor(
        X: number[][] = [],
        leafSize: number = 40,
        metric: Distance.IDistanceType = 'euclidiean',
        p: number = 2
    ) {
        this.X = X;
        this.leafSize = leafSize;
        this.distance = Distance.useDistance(metric);
        this.p = p;
        this.root = null;
        if (X.length > 0) {
            this.root = this.buildTree(Array.from({ length: X.length }, (_, i) => i));
        }
    }

    public fit(X: number[][]): void {
        this.X = X;
        this.root = this.buildTree(Array.from({ length: X.length }, (_, i) => i));
    }

    public query(X: number[][], k: number = 1): { distances: number[][]; indices: number[][] } {
        return {
            distances: X.map((x) => {
                const heap: Array<{ dist: number; idx: number }> = [];
                if (this.root) this._query(this.root, x, k, heap);
                return heap.map((h) => h.dist);
            }),
            indices: X.map((x) => {
                const heap: Array<{ dist: number; idx: number }> = [];
                if (this.root) this._query(this.root, x, k, heap);
                return heap.map((h) => h.idx);
            }),
        };
    }

    public queryRadius(
        X: number[][],
        r: number,
        returnDistance: boolean = false
    ): number[][] | { indices: number[][]; distances: number[][] } {
        const ind: number[][] = [];
        const dist: number[][] = [];
        for (const x of X) {
            const iArr: number[] = [];
            const dArr: number[] = [];
            if (this.root) this._queryRadius(this.root, x, r, iArr, dArr, returnDistance);
            ind.push(iArr);
            dist.push(dArr);
        }
        if (returnDistance) {
            for (let i = 0; i < ind.length; i++) {
                const arr = ind[i].map((idx, j) => ({ idx, dist: dist[i][j] }));
                arr.sort((a, b) => a.dist - b.dist);
                ind[i] = arr.map((a) => a.idx);
                dist[i] = arr.map((a) => a.dist);
            }
            return { indices: ind, distances: dist };
        }
        return ind;
    }

    private _query(node: BallNode, x: number[], k: number, heap: Array<{ dist: number; idx: number }>): void {
        if (!node.left && !node.right) {
            for (const idx of node.indices) {
                const d = this.distance(x, this.X[idx], this.p);
                this.pushHeap(heap, { dist: d, idx }, k);
            }
            return;
        }
        if (node.left) {
            const leftBound = Math.max(
                0,
                this.distance(x, node.left.centroid, this.p) - node.left.radius
            );
            const rightBound = node.right
                ? Math.max(0, this.distance(x, node.right.centroid, this.p) - node.right.radius)
                : Infinity;
            if (leftBound < rightBound) {
                this._query(node.left, x, k, heap);
                if (node.right && (heap.length < k || rightBound <= heap[heap.length - 1].dist)) {
                    this._query(node.right, x, k, heap);
                }
            } else {
                if (node.right) this._query(node.right, x, k, heap);
                if (heap.length < k || leftBound <= heap[heap.length - 1].dist) {
                    this._query(node.left, x, k, heap);
                }
            }
        }
    }

    private _queryRadius(
        node: BallNode,
        x: number[],
        r: number,
        ind: number[],
        dist: number[],
        keepDist: boolean
    ): void {
        const dCentroid = this.distance(x, node.centroid, this.p);
        if (dCentroid - node.radius > r) return;
        if (!node.left && !node.right) {
            for (const idx of node.indices) {
                const d = this.distance(x, this.X[idx], this.p);
                if (d <= r) {
                    ind.push(idx);
                    if (keepDist) dist.push(d);
                }
            }
            return;
        }
        if (node.left) this._queryRadius(node.left, x, r, ind, dist, keepDist);
        if (node.right) this._queryRadius(node.right, x, r, ind, dist, keepDist);
    }

    private pushHeap(
        heap: Array<{ dist: number; idx: number }>,
        item: { dist: number; idx: number },
        k: number
    ): void {
        let pos = 0;
        while (pos < heap.length && heap[pos].dist < item.dist) pos++;
        heap.splice(pos, 0, item);
        if (heap.length > k) heap.pop();
    }

    private buildTree(indices: number[]): BallNode {
        const centroid = new Array(this.X[0].length).fill(0);
        for (const i of indices) {
            const row = this.X[i];
            for (let d = 0; d < row.length; d++) centroid[d] += row[d];
        }
        for (let d = 0; d < centroid.length; d++) centroid[d] /= indices.length;

        let radius = 0;
        for (const i of indices) {
            const d = this.distance(centroid, this.X[i], this.p);
            if (d > radius) radius = d;
        }
        if (indices.length <= this.leafSize) {
            return { centroid, radius, indices, left: null, right: null };
        }
        let bestDim = 0;
        let bestVar = -Infinity;
        for (let d = 0; d < centroid.length; d++) {
            let mean = 0;
            for (const i of indices) mean += this.X[i][d];
            mean /= indices.length;
            let v = 0;
            for (const i of indices) v += (this.X[i][d] - mean) ** 2;
            if (v > bestVar) {
                bestVar = v;
                bestDim = d;
            }
        }
        indices.sort((a, b) => this.X[a][bestDim] - this.X[b][bestDim]);
        const mid = Math.floor(indices.length / 2);
        const leftIdx = indices.slice(0, mid);
        const rightIdx = indices.slice(mid);
        const left = this.buildTree(leftIdx);
        const right = this.buildTree(rightIdx);
        return { centroid, radius, indices: [], left, right };
    }
}
