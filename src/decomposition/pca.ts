export class PCA {
    private nComponents: number | null;
    private components: number[][];
    private mean: number[];
    private explainedVariance: number[];

    constructor(nComponents: number | null = null) {
        this.nComponents = nComponents;
        this.components = [];
        this.mean = [];
        this.explainedVariance = [];
    }

    private static dot(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static matVecMul(A: number[][], v: number[]): number[] {
        return A.map(row => PCA.dot(row, v));
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
        const norm = Math.sqrt(PCA.dot(v, v));
        return v.map(x => x / norm);
    }

    private static powerIteration(A: number[][], iter: number = 100): {value: number, vector: number[]} {
        // use deterministic initialization to avoid sign flips between runs
        let v: number[] = new Array(A.length).fill(1);
        v = PCA.normalize(v);
        for (let i = 0; i < iter; i++) {
            const Av = PCA.matVecMul(A, v);
            v = PCA.normalize(Av);
        }
        const Av = PCA.matVecMul(A, v);
        const value = PCA.dot(v, Av);
        return { value, vector: v };
    }

    private static cloneMatrix(A: number[][]): number[][] {
        return A.map(r => r.slice());
    }

    public fit(X: number[][]): void {
        const nSamples = X.length;
        const nFeatures = X[0].length;
        this.mean = new Array(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nFeatures; j++) {
                this.mean[j] += X[i][j];
            }
        }
        for (let j = 0; j < nFeatures; j++) {
            this.mean[j] /= nSamples;
        }
        const Xc = X.map(row => row.map((v, j) => v - this.mean[j]));
        const cov: number[][] = [];
        for (let i = 0; i < nFeatures; i++) {
            cov.push(new Array(nFeatures).fill(0));
        }
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nFeatures; j++) {
                for (let k = 0; k < nFeatures; k++) {
                    cov[j][k] += Xc[i][j] * Xc[i][k];
                }
            }
        }
        for (let j = 0; j < nFeatures; j++) {
            for (let k = 0; k < nFeatures; k++) {
                cov[j][k] /= (nSamples - 1);
            }
        }
        const k = this.nComponents === null ? nFeatures : Math.min(this.nComponents, nFeatures);
        let A = PCA.cloneMatrix(cov);
        this.components = [];
        this.explainedVariance = [];
        for (let c = 0; c < k; c++) {
            const {value, vector} = PCA.powerIteration(A, 200);
            // ensure deterministic sign by forcing the largest absolute component to be positive
            let maxIdx = 0;
            for (let i = 1; i < vector.length; i++) {
                if (Math.abs(vector[i]) > Math.abs(vector[maxIdx])) {
                    maxIdx = i;
                }
            }
            if (vector[maxIdx] < 0) {
                for (let i = 0; i < vector.length; i++) {
                    vector[i] = -vector[i];
                }
            }
            this.components.push(vector.slice());
            this.explainedVariance.push(value);
            // deflate
            const outer = PCA.outer(vector, vector);
            for (let i = 0; i < nFeatures; i++) {
                for (let j = 0; j < nFeatures; j++) {
                    A[i][j] -= value * outer[i][j];
                }
            }
        }
    }

    public transform(X: number[][]): number[][] {
        const Xc = X.map(row => row.map((v, j) => v - this.mean[j]));
        const componentsT = this.components; // components are stored as vectors
        return Xc.map(row => componentsT.map(comp => PCA.dot(row, comp)));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): number[][] {
        return X.map(row => {
            const orig = new Array(this.mean.length).fill(0);
            for (let i = 0; i < this.components.length; i++) {
                for (let j = 0; j < orig.length; j++) {
                    orig[j] += this.components[i][j] * row[i];
                }
            }
            for (let j = 0; j < orig.length; j++) {
                orig[j] += this.mean[j];
            }
            return orig;
        });
    }

    public getComponents(): number[][] {
        return this.components;
    }

    public getMean(): number[] {
        return this.mean;
    }

    public getExplainedVariance(): number[] {
        return this.explainedVariance;
    }
}
