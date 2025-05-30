export interface MDSOptions {
    nComponents?: number;
    dissimilarity?: 'euclidean' | 'precomputed';
}

export class MDS {
    private nComponents: number;
    private dissimilarity: 'euclidean' | 'precomputed';
    private embedding: number[][];

    constructor(options: MDSOptions = {}) {
        const { nComponents = 2, dissimilarity = 'euclidean' } = options;
        this.nComponents = nComponents;
        this.dissimilarity = dissimilarity;
        this.embedding = [];
    }

    private static pairwiseDist(X: number[][]): number[][] {
        const n = X.length;
        const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < i; j++) {
                let s = 0;
                for (let k = 0; k < X[i].length; k++) {
                    const diff = X[i][k] - X[j][k];
                    s += diff * diff;
                }
                const d = Math.sqrt(s);
                D[i][j] = d;
                D[j][i] = d;
            }
        }
        return D;
    }

    private static dot(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static matVecMul(A: number[][], v: number[]): number[] {
        return A.map(row => MDS.dot(row, v));
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
        const norm = Math.sqrt(MDS.dot(v, v));
        return v.map(x => x / norm);
    }

    private static powerIteration(A: number[][], iter: number = 100): { value: number; vector: number[] } {
        let v: number[] = Array(A.length)
            .fill(0)
            .map(() => Math.random());
        v = MDS.normalize(v);
        for (let i = 0; i < iter; i++) {
            const Av = MDS.matVecMul(A, v);
            v = MDS.normalize(Av);
        }
        const Av = MDS.matVecMul(A, v);
        const value = MDS.dot(v, Av);
        return { value, vector: v };
    }

    private static cloneMatrix(A: number[][]): number[][] {
        return A.map(r => r.slice());
    }

    public fitTransform(data: number[][]): number[][] {
        const D = this.dissimilarity === 'precomputed' ? data : MDS.pairwiseDist(data);
        const n = D.length;
        const D2: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                D2[i][j] = D[i][j] ** 2;
            }
        }
        const rowMean = new Array(n).fill(0);
        const colMean = new Array(n).fill(0);
        let totalMean = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                rowMean[i] += D2[i][j];
                colMean[j] += D2[i][j];
                totalMean += D2[i][j];
            }
        }
        for (let i = 0; i < n; i++) rowMean[i] /= n;
        for (let j = 0; j < n; j++) colMean[j] /= n;
        totalMean /= n * n;

        const B: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                B[i][j] = -0.5 * (D2[i][j] - rowMean[i] - colMean[j] + totalMean);
            }
        }

        let A = MDS.cloneMatrix(B);
        const vectors: number[][] = [];
        const values: number[] = [];
        const k = Math.min(this.nComponents, n);
        for (let c = 0; c < k; c++) {
            const { value, vector } = MDS.powerIteration(A, 200);
            vectors.push(vector.slice());
            values.push(value);
            const outer = MDS.outer(vector, vector);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    A[i][j] -= value * outer[i][j];
                }
            }
        }

        const result: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
        for (let j = 0; j < k; j++) {
            const scale = Math.sqrt(Math.max(values[j], 0));
            for (let i = 0; i < n; i++) {
                result[i][j] = vectors[j][i] * scale;
            }
        }
        this.embedding = result;
        return result;
    }

    public getEmbedding(): number[][] {
        return this.embedding;
    }
}
