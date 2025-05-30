export interface TSNEOptions {
    nComponents?: number;
    perplexity?: number;
    learningRate?: number;
    nIter?: number;
}

export class TSNE {
    private nComponents: number;
    private perplexity: number;
    private learningRate: number;
    private nIter: number;
    private embedding: number[][] = [];

    constructor(options: TSNEOptions = {}) {
        this.nComponents = options.nComponents ?? 2;
        this.perplexity = options.perplexity ?? 30;
        this.learningRate = options.learningRate ?? 200;
        this.nIter = options.nIter ?? 250;
    }

    private static squaredDistance(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    private static randomMatrix(rows: number, cols: number): number[][] {
        const mat: number[][] = [];
        for (let i = 0; i < rows; i++) {
            mat.push([]);
            for (let j = 0; j < cols; j++) {
                mat[i][j] = (Math.random() - 0.5) * 1e-4;
            }
        }
        return mat;
    }

    private computeP(dist: number[][]): number[][] {
        const n = dist.length;
        const P: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));
        const logU = Math.log(this.perplexity);
        for (let i = 0; i < n; i++) {
            const Di = dist[i].slice();
            Di[i] = Infinity;
            let betamin = -Infinity;
            let betamax = Infinity;
            let beta = 1.0;
            for (let iter = 0; iter < 50; iter++) {
                const HBeta = this.Hbeta(Di, beta);
                const Hdiff = HBeta.H - logU;
                if (Math.abs(Hdiff) < 1e-5) break;
                if (Hdiff > 0) {
                    betamin = beta;
                    beta = betamax === Infinity ? beta * 2 : (beta + betamax) / 2;
                } else {
                    betamax = beta;
                    beta = betamin === -Infinity ? beta / 2 : (beta + betamin) / 2;
                }
            }
            const Pi = this.Hbeta(Di, beta).P;
            for (let j = 0; j < n; j++) {
                P[i][j] = i === j ? 0 : Pi[j];
            }
        }
        // symmetrize
        let sumP = 0;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const val = (P[i][j] + P[j][i]) / (2 * n);
                P[i][j] = P[j][i] = val;
                sumP += 2 * val;
            }
            P[i][i] = 0;
        }
        // normalize (should already sum to 1 but ensure numerical stability)
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                P[i][j] /= sumP;
            }
        }
        return P;
    }

    private Hbeta(Di: number[], beta: number): { H: number; P: number[] } {
        const P = Di.map((d) => Math.exp(-d * beta));
        const sumP = P.reduce((a, b) => a + b, 0);
        const Pi = P.map((p) => p / sumP);
        let H = 0;
        for (let i = 0; i < Pi.length; i++) {
            if (Pi[i] > 1e-7) {
                H -= Pi[i] * Math.log(Pi[i]);
            }
        }
        return { H, P: Pi };
    }

    public fit(X: number[][]): void {
        const n = X.length;
        const dist: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const d = TSNE.squaredDistance(X[i], X[j]);
                dist[i][j] = dist[j][i] = d;
            }
        }
        const P = this.computeP(dist);
        let Y = TSNE.randomMatrix(n, this.nComponents);
        let dY = new Array(n).fill(0).map(() => new Array(this.nComponents).fill(0));
        let iY = new Array(n).fill(0).map(() => new Array(this.nComponents).fill(0));
        let momentum = 0.5;
        for (let iter = 0; iter < this.nIter; iter++) {
            const { Q, num } = this.computeQ(Y);
            dY = dY.map(row => row.fill(0));
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    const mult = (P[i][j] - Q[i][j]) * num[i][j];
                    for (let d = 0; d < this.nComponents; d++) {
                        dY[i][d] += mult * (Y[i][d] - Y[j][d]);
                    }
                }
            }
            for (let i = 0; i < n; i++) {
                for (let d = 0; d < this.nComponents; d++) {
                    iY[i][d] = momentum * iY[i][d] + this.learningRate * dY[i][d] * 4;
                    Y[i][d] += iY[i][d];
                }
            }
            if (iter === 100) momentum = 0.8;
        }
        this.embedding = Y;
    }

    private computeQ(Y: number[][]): { Q: number[][]; num: number[][] } {
        const n = Y.length;
        const Q: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));
        const num: number[][] = new Array(n).fill(0).map(() => new Array(n).fill(0));
        let sum = 0;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const dist = TSNE.squaredDistance(Y[i], Y[j]);
                const v = 1 / (1 + dist);
                num[i][j] = num[j][i] = v;
                sum += 2 * v;
            }
        }
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const val = num[i][j] / sum;
                Q[i][j] = Q[j][i] = val;
            }
            Q[i][i] = 0;
        }
        return { Q, num };
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.embedding;
    }

    public getEmbedding(): number[][] {
        return this.embedding;
    }
}
