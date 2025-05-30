export interface SpectralEmbeddingProps {
    nComponents?: number;
    nNeighbors?: number;
}

export class SpectralEmbedding {
    private nComponents: number;
    private nNeighbors: number;
    private embedding: number[][];

    constructor(props: SpectralEmbeddingProps = {}) {
        const { nComponents = 2, nNeighbors = 10 } = props;
        this.nComponents = nComponents;
        this.nNeighbors = nNeighbors;
        this.embedding = [];
    }

    private static dot(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static matVecMul(A: number[][], v: number[]): number[] {
        return A.map(row => SpectralEmbedding.dot(row, v));
    }

    private static outer(v1: number[], v2: number[]): number[][] {
        const res: number[][] = [];
        for (let i = 0; i < v1.length; i++) {
            res.push([]);
            for (let j = 0; j < v2.length; j++) {
                res[i].push(v1[i] * v2[j]);
            }
        }
        return res;
    }

    private static normalize(v: number[]): number[] {
        const norm = Math.sqrt(SpectralEmbedding.dot(v, v));
        return v.map(x => x / norm);
    }

    private static powerIteration(A: number[][], iter: number = 500): {value: number, vector: number[]} {
        let v: number[] = Array(A.length).fill(1).map(() => Math.random());
        v = SpectralEmbedding.normalize(v);
        for (let i = 0; i < iter; i++) {
            const Av = SpectralEmbedding.matVecMul(A, v);
            v = SpectralEmbedding.normalize(Av);
        }
        const Av = SpectralEmbedding.matVecMul(A, v);
        const value = SpectralEmbedding.dot(v, Av);
        return { value, vector: v };
    }

    private static signFlip(v: number[]): number[] {
        let idx = 0;
        for (let i = 1; i < v.length; i++) {
            if (Math.abs(v[i]) > Math.abs(v[idx])) idx = i;
        }
        if (v[idx] < 0) return v.map(x => -x);
        return v;
    }

    private static cloneMatrix(A: number[][]): number[][] {
        return A.map(r => r.slice());
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
        const k = this.nComponents + 1;
        let B = SpectralEmbedding.cloneMatrix(A);
        const comps: number[][] = [];
        for (let c = 0; c < k; c++) {
            let { value, vector } = SpectralEmbedding.powerIteration(B, 1000);
            for (let v of comps) {
                const proj = SpectralEmbedding.dot(vector, v);
                for (let i = 0; i < vector.length; i++) vector[i] -= proj * v[i];
            }
            vector = SpectralEmbedding.normalize(vector);
            comps.push(SpectralEmbedding.signFlip(vector.slice()));
            const outer = SpectralEmbedding.outer(vector, vector);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    B[i][j] -= value * outer[i][j];
                }
            }
        }
        const selected = comps.slice(1, k);
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
