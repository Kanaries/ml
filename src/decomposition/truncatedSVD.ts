import { dot, symmetricEigen } from '../algebra/eigen';

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
        const { values, vectors } = symmetricEigen(XtX, k);
        this.components = [];
        this.singularValues = [];
        this.explainedVariance = [];

        for (let c = 0; c < vectors.length; c++) {
            const vector = vectors[c].slice();

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

            this.components.push(vector);
            const s = Math.sqrt(Math.max(values[c], 0));
            this.singularValues.push(s);
        }

        // sklearn definition: explained_variance_[i] = Var(X_transformed[:, i])
        // (mean-subtracted, ddof=0) — NOT s^2/(n-1); X is not centered here, so
        // the score columns have non-zero means that must be subtracted.
        const scores = this.transform(X);
        this.explainedVariance = this.components.map((_, c) => {
            let mu = 0;
            for (let i = 0; i < nSamples; i++) mu += scores[i][c];
            mu /= nSamples;
            let v = 0;
            for (let i = 0; i < nSamples; i++) v += (scores[i][c] - mu) ** 2;
            return v / nSamples;
        });

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
        return X.map(row => this.components.map(comp => dot(row, comp)));
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
