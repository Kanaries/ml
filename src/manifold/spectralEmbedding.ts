import { BaseEstimator, registerEstimator, Params } from '../base/estimator';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';
import { validateMatrix } from '../utils/numerics';

export interface SpectralEmbeddingProps {
    nComponents?: number;
    nNeighbors?: number;
    randomState?: number;
}

// embedding-only estimator (no out-of-sample transform): extends BaseEstimator
export class SpectralEmbedding extends BaseEstimator {
    private nComponents: number;
    private nNeighbors: number;
    private randomState?: number;
    private embedding: number[][];

    constructor(props: SpectralEmbeddingProps = {}) {
        super();
        const { nComponents = 2, nNeighbors = 10, randomState } = props;
        if (!Number.isInteger(nComponents) || nComponents < 1 || !Number.isInteger(nNeighbors) || nNeighbors < 1) throw new Error('nComponents and nNeighbors must be positive integers');
        this.nComponents = nComponents;
        this.nNeighbors = nNeighbors;
        this.randomState = randomState;
        this.embedding = [];
    }

    public getParams(): Params {
        return {
            nComponents: this.nComponents,
            nNeighbors: this.nNeighbors,
            randomState: this.randomState,
        };
    }

    private static signFlip(v: number[]): number[] {
        let idx = 0;
        for (let i = 1; i < v.length; i++) {
            if (Math.abs(v[i]) > Math.abs(v[idx])) idx = i;
        }
        if (v[idx] < 0) return v.map(x => -x);
        return v;
    }

    private euclidean(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
        return Math.sqrt(s);
    }

    private constructAffinity(X: number[][]): number[][] {
        const n = X.length;
        const W = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            W[i][i] = 1;
        }
        for (let i = 0; i < n; i++) {
            const dists = X.map((row, j) => ({ j, d: this.euclidean(X[i], row) }));
            dists.sort((a, b) => a.d - b.d);
            for (let k = 1; k <= Math.min(this.nNeighbors, n - 1); k++) {
                const idx = dists[k].j;
                W[i][idx] = 1;
                W[idx][i] = 1;
            }
        }
        return W;
    }

    public fit(X: number[][]): void {
        validateMatrix(X, 2);
        if (this.nComponents >= X.length) throw new Error('nComponents must be less than the number of samples');
        const W = this.constructAffinity(X);
        const n = W.length;
        const D = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            D[i] = W[i].reduce((a, b) => a + b, 0);
        }
        const Dn = D.map(d => (d === 0 ? 0 : 1 / Math.sqrt(d)));
        const A: number[][] = [];
        for (let i = 0; i < n; i++) {
            A.push(new Array(n).fill(0));
            for (let j = 0; j < n; j++) {
                A[i][j] = W[i][j] * Dn[i] * Dn[j];
            }
        }
        // Smallest normalized-Laplacian modes correspond to the largest
        // algebraic eigenvalues of A. Negative eigenvalues must not outrank
        // positive ones merely because their absolute values are larger.
        const eigen = symmetricEigDecomposition(A);
        const selected = eigen.vectors.slice(1, this.nComponents + 1).map(SpectralEmbedding.signFlip);
        this.embedding = Array.from({ length: n }, () => new Array(this.nComponents).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < selected.length; j++) {
                this.embedding[i][j] = selected[j][i] * Dn[i];
            }
        }
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.embedding;
    }

    public getEmbedding(): number[][] {
        return this.embedding;
    }
}
registerEstimator('SpectralEmbedding', SpectralEmbedding);
