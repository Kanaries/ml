import { Inverse } from '../algebra';

export class LocallyLinearEmbedding {
    private nNeighbors: number;
    private nComponents: number;
    private reg: number;
    private trainX: number[][] = [];
    private trainY: number[][] = [];

    constructor(nNeighbors: number = 5, nComponents: number = 2, reg: number = 0.001) {
        this.nNeighbors = nNeighbors;
        this.nComponents = nComponents;
        this.reg = reg;
    }

    private static dot(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static matVecMul(A: number[][], v: number[]): number[] {
        return A.map(r => LocallyLinearEmbedding.dot(r, v));
    }

    private static normalize(v: number[]): number[] {
        const n = Math.sqrt(LocallyLinearEmbedding.dot(v, v));
        return v.map(x => x / n);
    }

    private static outer(v1: number[], v2: number[]): number[][] {
        const res: number[][] = [];
        for (let i = 0; i < v1.length; i++) {
            res.push([]);
            for (let j = 0; j < v2.length; j++) res[i].push(v1[i] * v2[j]);
        }
        return res;
    }

    private static cloneMatrix(A: number[][]): number[][] {
        return A.map(r => r.slice());
    }

    private static powerIteration(A: number[][], iter: number = 100): { value: number; vector: number[] } {
        let v: number[] = Array(A.length).fill(0).map(() => Math.random());
        v = LocallyLinearEmbedding.normalize(v);
        for (let i = 0; i < iter; i++) {
            const Av = LocallyLinearEmbedding.matVecMul(A, v);
            v = LocallyLinearEmbedding.normalize(Av);
        }
        const Av = LocallyLinearEmbedding.matVecMul(A, v);
        const value = LocallyLinearEmbedding.dot(v, Av);
        return { value, vector: v };
    }

    private distance(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
        return Math.sqrt(s);
    }

    private neighbors(x: number[], X: number[][], exclude: number | null = null): number[] {
        const dists: Array<{ idx: number; dis: number }> = [];
        for (let i = 0; i < X.length; i++) {
            if (exclude !== null && i === exclude) continue;
            dists.push({ idx: i, dis: this.distance(x, X[i]) });
        }
        dists.sort((a, b) => a.dis - b.dis);
        return dists.slice(0, Math.min(this.nNeighbors, dists.length)).map(d => d.idx);
    }

    private computeWeights(x: number[], neighborIdx: number[]): number[] {
        const k = neighborIdx.length;
        const Z: number[][] = [];
        for (let i = 0; i < k; i++) {
            const diff = [] as number[];
            for (let j = 0; j < x.length; j++) diff.push(this.trainX[neighborIdx[i]][j] - x[j]);
            Z.push(diff);
        }
        const G: number[][] = [];
        for (let i = 0; i < k; i++) {
            G.push(new Array(k).fill(0));
        }
        for (let i = 0; i < k; i++) {
            for (let j = i; j < k; j++) {
                const val = LocallyLinearEmbedding.dot(Z[i], Z[j]);
                G[i][j] = val;
                G[j][i] = val;
            }
        }
        let trace = 0;
        for (let i = 0; i < k; i++) trace += G[i][i];
        const eps = this.reg * trace;
        for (let i = 0; i < k; i++) G[i][i] += eps;
        let inv = Inverse.elementary(G);
        if (inv === false) {
            for (let i = 0; i < k; i++) G[i][i] += this.reg;
            inv = Inverse.elementary(G);
            if (inv === false) throw new Error('matrix not invertible');
        }
        const ones = new Array(k).fill(1);
        const w = (inv as number[][]).map(row => LocallyLinearEmbedding.dot(row, ones));
        let s = 0;
        for (let i = 0; i < k; i++) s += w[i];
        return w.map(v => v / s);
    }

    public fit(X: number[][]): void {
        this.trainX = X;
        const n = X.length;
        const W: number[][] = [];
        for (let i = 0; i < n; i++) W.push(new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            const nbrs = this.neighbors(X[i], X, i);
            const w = this.computeWeights(X[i], nbrs);
            for (let j = 0; j < nbrs.length; j++) {
                W[i][nbrs[j]] = w[j];
            }
        }
        const B: number[][] = [];
        for (let i = 0; i < n; i++) {
            B.push(new Array(n).fill(0));
            for (let j = 0; j < n; j++) {
                B[i][j] = -W[i][j];
            }
            B[i][i] += 1;
        }
        const M: number[][] = [];
        for (let i = 0; i < n; i++) M.push(new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) sum += B[k][i] * B[k][j];
                M[i][j] = sum;
            }
        }
        for (let i = 0; i < n; i++) M[i][i] += this.reg;
        let invM = Inverse.elementary(M);
        if (invM === false) throw new Error('failed to invert matrix');
        const A = LocallyLinearEmbedding.cloneMatrix(invM as number[][]);
        const eigenVecs: number[][] = [];
        for (let c = 0; c < this.nComponents + 1; c++) {
            const { value, vector } = LocallyLinearEmbedding.powerIteration(A, 200);
            eigenVecs.push(vector.slice());
            const outer = LocallyLinearEmbedding.outer(vector, vector);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    A[i][j] -= value * outer[i][j];
                }
            }
        }
        const Y: number[][] = [];
        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            for (let c = 1; c < this.nComponents + 1; c++) {
                row.push(eigenVecs[c][i]);
            }
            Y.push(row);
        }
        this.trainY = Y;
    }

    public transform(X: number[][]): number[][] {
        const result: number[][] = [];
        for (const x of X) {
            const nbrs = this.neighbors(x, this.trainX);
            const w = this.computeWeights(x, nbrs);
            const row = new Array(this.nComponents).fill(0);
            for (let i = 0; i < nbrs.length; i++) {
                for (let c = 0; c < this.nComponents; c++) {
                    row[c] += w[i] * this.trainY[nbrs[i]][c];
                }
            }
            result.push(row);
        }
        return result;
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.trainY;
    }
}
