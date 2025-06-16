import { PCA } from './pca';

export interface SparsePCAProps {
    nComponents?: number | null;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}

export class SparsePCA extends PCA {
    private alpha: number;
    private maxIter: number;
    private tol: number;

    constructor(props: SparsePCAProps = {}) {
        const { nComponents = null, alpha = 1, maxIter = 100, tol = 1e-8 } = props;
        super(nComponents);
        this.alpha = alpha;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public fit(X: number[][]): void {
        if (this.alpha === 0) {
            super.fit(X);
            return;
        }
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
        let cov: number[][] = [];
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
            let v: number[] = new Array(nFeatures).fill(1 / Math.sqrt(nFeatures));
            for (let iter = 0; iter < this.maxIter; iter++) {
                const oldV = v.slice();
                let Av = PCA.matVecMul(A, v);
                for (let i = 0; i < Av.length; i++) {
                    const sign = Av[i] >= 0 ? 1 : -1;
                    const val = Math.max(Math.abs(Av[i]) - this.alpha, 0);
                    Av[i] = sign * val;
                }
                const norm = Math.sqrt(PCA.dot(Av, Av));
                if (norm === 0) {
                    break;
                }
                v = Av.map(x => x / norm);
                const diff = Math.sqrt(v.reduce((s, val, i) => s + (val - oldV[i]) ** 2, 0));
                if (diff < this.tol) {
                    break;
                }
            }
            const value = PCA.dot(v, PCA.matVecMul(cov, v));
            // ensure deterministic sign
            let maxIdx = 0;
            for (let i = 1; i < v.length; i++) {
                if (Math.abs(v[i]) > Math.abs(v[maxIdx])) {
                    maxIdx = i;
                }
            }
            if (v[maxIdx] < 0) {
                for (let i = 0; i < v.length; i++) {
                    v[i] = -v[i];
                }
            }
            this.components.push(v.slice());
            this.explainedVariance.push(value);
            const outer = PCA.outer(v, v);
            for (let i = 0; i < nFeatures; i++) {
                for (let j = 0; j < nFeatures; j++) {
                    A[i][j] -= value * outer[i][j];
                }
            }
        }
    }
}
