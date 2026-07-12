import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { KMeans } from './kmeans';

/**
 * SpectralClustering, following sklearn.cluster.SpectralClustering.
 *
 * 1. Build the affinity matrix W:
 *    - 'rbf':  W_ij = exp(-gamma * ||x_i - x_j||^2)
 *    - 'nearestNeighbors': k-nearest-neighbor connectivity graph (each
 *      sample counts itself among its neighbors, as sklearn's
 *      kneighbors_graph(include_self=True) does), symmetrized as
 *      W = 0.5 * (A + A^T) exactly like sklearn.
 * 2. Form the symmetric normalized Laplacian
 *    L_sym = I - D^{-1/2} W D^{-1/2} (degree-0 nodes are left untouched,
 *    like scipy.sparse.csgraph.laplacian(normed=True)).
 * 3. Take the eigenvectors of the `nClusters` smallest eigenvalues of
 *    L_sym. The full symmetric eigendecomposition is computed with the
 *    cyclic Jacobi rotation method — deterministic and robust for the
 *    (near-)degenerate zero eigenvalues that connected components produce,
 *    where power iteration converges poorly.
 * 4. Row-normalize the spectral embedding (Ng, Jordan & Weiss 2001) and
 *    cluster the rows with seeded KMeans (`nInit` restarts).
 *
 * Only assignLabels='kmeans' is supported (sklearn additionally offers
 * 'discretize' and 'cluster_qr').
 */

export type SpectralAffinity = 'rbf' | 'nearestNeighbors';

export interface SpectralClusteringProps {
    /** number of clusters / dimension of the spectral embedding */
    nClusters?: number;
    /** how to construct the affinity matrix */
    affinity?: SpectralAffinity;
    /** kernel coefficient for the rbf affinity */
    gamma?: number;
    /** number of neighbors for the nearestNeighbors affinity */
    nNeighbors?: number;
    /** seed for the KMeans step */
    randomState?: number;
    /** number of KMeans restarts on the embedding */
    nInit?: number;
    /** label assignment strategy in the embedding space */
    assignLabels?: 'kmeans';
}

/**
 * Full eigendecomposition of a symmetric matrix by the cyclic Jacobi
 * rotation method. Returns eigenvalues in ascending order with the matching
 * eigenvectors (each a length-n array). Deterministic; handles repeated
 * eigenvalues (the rotations always produce an orthonormal basis).
 */
export function jacobiEigenSymmetric(
    A: number[][],
    maxSweeps: number = 100,
    tol: number = 1e-12
): { values: number[]; vectors: number[][] } {
    const n = A.length;
    const a = A.map((row) => row.slice());
    // V columns are the eigenvectors: V starts as the identity
    const V: number[][] = [];
    for (let i = 0; i < n; i++) {
        const row = new Array(n).fill(0);
        row[i] = 1;
        V.push(row);
    }
    let scale = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) scale = Math.max(scale, Math.abs(a[i][j]));
    }
    const threshold = tol * (scale || 1);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
        let off = 0;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) off = Math.max(off, Math.abs(a[i][j]));
        }
        if (off <= threshold) break;
        for (let p = 0; p < n - 1; p++) {
            for (let q = p + 1; q < n; q++) {
                const apq = a[p][q];
                if (Math.abs(apq) <= threshold) continue;
                const theta = (a[q][q] - a[p][p]) / (2 * apq);
                const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                const c = 1 / Math.sqrt(t * t + 1);
                const s = t * c;
                // rotate columns p and q
                for (let i = 0; i < n; i++) {
                    const aip = a[i][p];
                    const aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[i][q] = s * aip + c * aiq;
                }
                // rotate rows p and q
                for (let i = 0; i < n; i++) {
                    const api = a[p][i];
                    const aqi = a[q][i];
                    a[p][i] = c * api - s * aqi;
                    a[q][i] = s * api + c * aqi;
                }
                // accumulate the rotation into the eigenvector matrix
                for (let i = 0; i < n; i++) {
                    const vip = V[i][p];
                    const viq = V[i][q];
                    V[i][p] = c * vip - s * viq;
                    V[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
    const values = order.map((i) => a[i][i]);
    const vectors = order.map((col) => V.map((row) => row[col]));
    return { values, vectors };
}

export class SpectralClustering extends ClusterBase {
    private nClusters: number;
    private affinity: SpectralAffinity;
    private gamma: number;
    private nNeighbors: number;
    private randomState?: number;
    private nInit: number;
    private assignLabels: 'kmeans';
    private labels: number[] = [];

    constructor(props: SpectralClusteringProps = {}) {
        super();
        const {
            nClusters = 8,
            affinity = 'rbf',
            gamma = 1,
            nNeighbors = 10,
            randomState,
            nInit = 10,
            assignLabels = 'kmeans',
        } = props;
        if (affinity !== 'rbf' && affinity !== 'nearestNeighbors') {
            throw new Error(`Unknown affinity "${affinity}"; expected 'rbf' or 'nearestNeighbors'`);
        }
        if (assignLabels !== 'kmeans') {
            throw new Error(`assignLabels="${assignLabels}" is not supported; only 'kmeans' is implemented`);
        }
        this.nClusters = nClusters;
        this.affinity = affinity;
        this.gamma = gamma;
        this.nNeighbors = nNeighbors;
        this.randomState = randomState;
        this.nInit = nInit;
        this.assignLabels = assignLabels;
    }

    public getParams(): Params {
        return {
            nClusters: this.nClusters,
            affinity: this.affinity,
            gamma: this.gamma,
            nNeighbors: this.nNeighbors,
            randomState: this.randomState,
            nInit: this.nInit,
            assignLabels: this.assignLabels,
        };
    }

    public fitPredict(samplesX: number[][]): number[] {
        const n = samplesX.length;
        this.labels = [];
        if (n === 0) return [];
        if (this.nClusters > n) {
            throw new Error(`Cannot extract ${this.nClusters} clusters from ${n} samples`);
        }

        const W = this.affinityMatrix(samplesX);

        // symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2};
        // isolated nodes (degree 0) get d^{-1/2} = 1, matching scipy's
        // csgraph.laplacian(normed=True)
        const invSqrtDeg: number[] = new Array(n);
        for (let i = 0; i < n; i++) {
            let deg = 0;
            for (let j = 0; j < n; j++) deg += W[i][j];
            invSqrtDeg[i] = deg > 0 ? 1 / Math.sqrt(deg) : 1;
        }
        const L: number[][] = new Array(n);
        for (let i = 0; i < n; i++) {
            L[i] = new Array(n);
            for (let j = 0; j < n; j++) {
                L[i][j] = (i === j ? 1 : 0) - invSqrtDeg[i] * W[i][j] * invSqrtDeg[j];
            }
        }

        // eigenvectors of the nClusters smallest eigenvalues
        const { vectors } = jacobiEigenSymmetric(L);
        const k = this.nClusters;
        const embedding: number[][] = new Array(n);
        for (let i = 0; i < n; i++) {
            const row = new Array(k);
            for (let c = 0; c < k; c++) row[c] = vectors[c][i];
            embedding[i] = row;
        }
        // Ng-Jordan-Weiss row normalization
        for (let i = 0; i < n; i++) {
            let norm = 0;
            for (let c = 0; c < k; c++) norm += embedding[i][c] ** 2;
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (let c = 0; c < k; c++) embedding[i][c] /= norm;
            }
        }

        const kmeans = new KMeans({
            n_clusters: k,
            random_state: this.randomState,
            n_init: this.nInit,
            max_iter: 300,
        });
        this.labels = kmeans.fitPredict(embedding);
        return this.labels;
    }

    public getLabels(): number[] {
        return this.labels;
    }

    private affinityMatrix(samplesX: number[][]): number[][] {
        const n = samplesX.length;
        const sqDist = (a: number[], b: number[]): number => {
            let s = 0;
            for (let d = 0; d < a.length; d++) {
                const diff = a[d] - b[d];
                s += diff * diff;
            }
            return s;
        };
        if (this.affinity === 'rbf') {
            const W: number[][] = new Array(n);
            for (let i = 0; i < n; i++) W[i] = new Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                W[i][i] = 1;
                for (let j = i + 1; j < n; j++) {
                    const w = Math.exp(-this.gamma * sqDist(samplesX[i], samplesX[j]));
                    W[i][j] = w;
                    W[j][i] = w;
                }
            }
            return W;
        }
        // nearestNeighbors: 0/1 kNN connectivity (self included among the
        // neighbors, sklearn's include_self=True), symmetrized to
        // 0.5 * (A + A^T) like sklearn
        const kNeighbors = Math.min(this.nNeighbors, n);
        const A: number[][] = new Array(n);
        for (let i = 0; i < n; i++) A[i] = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            const dists: number[] = new Array(n);
            for (let j = 0; j < n; j++) dists[j] = sqDist(samplesX[i], samplesX[j]);
            const order = Array.from({ length: n }, (_, j) => j)
                .sort((a, b) => (dists[a] - dists[b]) || (a - b));
            for (let r = 0; r < kNeighbors; r++) {
                A[i][order[r]] = 1;
            }
        }
        const W: number[][] = new Array(n);
        for (let i = 0; i < n; i++) {
            W[i] = new Array(n);
            for (let j = 0; j < n; j++) {
                W[i][j] = 0.5 * (A[i][j] + A[j][i]);
            }
        }
        return W;
    }
}
registerEstimator('SpectralClustering', SpectralClustering);
