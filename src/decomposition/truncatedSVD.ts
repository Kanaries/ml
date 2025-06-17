export class TruncatedSVD {
    private nComponents: number;
    private components: number[][];
    private singularValues: number[];
    private explainedVariance: number[];
    private explainedVarianceRatio: number[];

    constructor(nComponents: number = 2) {
        this.nComponents = nComponents;
        this.components = [];
        this.singularValues = [];
        this.explainedVariance = [];
        this.explainedVarianceRatio = [];
    }

    private static dot(a: number[], b: number[]): number {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static matVecMul(A: number[][], v: number[]): number[] {
        return A.map(row => TruncatedSVD.dot(row, v));
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
        const norm = Math.sqrt(TruncatedSVD.dot(v, v));
        return v.map(x => x / norm);
    }

    private static powerIteration(A: number[][], iter: number = 100): { value: number; vector: number[] } {
        let v: number[] = new Array(A.length).fill(1);
        v = TruncatedSVD.normalize(v);
        for (let i = 0; i < iter; i++) {
            const Av = TruncatedSVD.matVecMul(A, v);
            v = TruncatedSVD.normalize(Av);
        }
        const Av = TruncatedSVD.matVecMul(A, v);
        const value = TruncatedSVD.dot(v, Av);
        return { value, vector: v };
    }

    private static cloneMatrix(A: number[][]): number[][] {
        return A.map(r => r.slice());
    }

    public fit(X: number[][]): void {
        const nSamples = X.length;
        const nFeatures = X[0].length;

        const XtX: number[][] = [];
        for (let i = 0; i < nFeatures; i++) {
            XtX.push(new Array(nFeatures).fill(0));
        }
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nFeatures; j++) {
                for (let k = 0; k < nFeatures; k++) {
                    XtX[j][k] += X[i][j] * X[i][k];
                }
            }
        }

        const k = Math.min(this.nComponents, nFeatures);
        let A = TruncatedSVD.cloneMatrix(XtX);
        this.components = [];
        this.singularValues = [];
        this.explainedVariance = [];

        for (let c = 0; c < k; c++) {
            const { value, vector } = TruncatedSVD.powerIteration(A, 200);

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
            const s = Math.sqrt(Math.max(value, 0));
            this.singularValues.push(s);
            this.explainedVariance.push((s * s) / (nSamples - 1));

            const outer = TruncatedSVD.outer(vector, vector);
            for (let i = 0; i < nFeatures; i++) {
                for (let j = 0; j < nFeatures; j++) {
                    A[i][j] -= value * outer[i][j];
                }
            }
        }

        const featureVar = new Array(nFeatures).fill(0);
        const mean = new Array(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nFeatures; j++) {
                mean[j] += X[i][j];
            }
        }
        for (let j = 0; j < nFeatures; j++) {
            mean[j] /= nSamples;
        }
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nFeatures; j++) {
                const diff = X[i][j] - mean[j];
                featureVar[j] += diff * diff;
            }
        }
        for (let j = 0; j < nFeatures; j++) {
            featureVar[j] /= nSamples;
        }
        const totalVar = featureVar.reduce((a, b) => a + b, 0);

        this.explainedVarianceRatio = this.explainedVariance.map(v => v / totalVar);
    }

    public transform(X: number[][]): number[][] {
        return X.map(row => this.components.map(comp => TruncatedSVD.dot(row, comp)));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): number[][] {
        return X.map(row => {
            const orig = new Array(this.components[0].length).fill(0);
            for (let i = 0; i < this.components.length; i++) {
                for (let j = 0; j < orig.length; j++) {
                    orig[j] += this.components[i][j] * row[i];
                }
            }
            return orig;
        });
    }

    public getComponents(): number[][] {
        return this.components;
    }

    public getSingularValues(): number[] {
        return this.singularValues;
    }

    public getExplainedVariance(): number[] {
        return this.explainedVariance;
    }

    public getExplainedVarianceRatio(): number[] {
        return this.explainedVarianceRatio;
    }
}
