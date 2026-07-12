import { dot, symmetricEigen } from '../algebra/eigen';
import { TransformerBase } from '../base/transformer';
import { registerEstimator, Params } from '../base/estimator';

export interface PCAProps {
    /** number of components to keep; null keeps all features */
    nComponents?: number | null;
}

export class PCA extends TransformerBase {
    protected nComponents: number | null;
    protected components: number[][];
    protected mean: number[];
    protected explainedVariance: number[];

    constructor(props?: PCAProps);
    /** @deprecated positional form; prefer the props-object constructor */
    constructor(nComponents?: number | null);
    constructor(arg0: PCAProps | number | null = {}) {
        super();
        let props: PCAProps;
        if (typeof arg0 === 'number') {
            props = { nComponents: arg0 };
        } else if (arg0 === null) {
            props = { nComponents: null };
        } else {
            props = arg0;
        }
        const { nComponents = null } = props;
        this.nComponents = nComponents;
        this.components = [];
        this.mean = [];
        this.explainedVariance = [];
    }

    public getParams(): Params {
        return { nComponents: this.nComponents };
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
        const { values, vectors } = symmetricEigen(cov, k);
        this.components = [];
        this.explainedVariance = [];
        for (let c = 0; c < vectors.length; c++) {
            const vector = vectors[c].slice();
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
            this.components.push(vector);
            // covariance is PSD: clamp tiny negative Rayleigh quotients from round-off
            this.explainedVariance.push(Math.max(values[c], 0));
        }
    }

    public transform(X: number[][]): number[][] {
        const Xc = X.map(row => row.map((v, j) => v - this.mean[j]));
        const componentsT = this.components; // components are stored as vectors
        return Xc.map(row => componentsT.map(comp => dot(row, comp)));
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
registerEstimator('PCA', PCA);
