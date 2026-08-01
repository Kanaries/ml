import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import { matMul, normalRandom, pseudoInverseSymmetric, transpose, validateMatrix } from '../utils/numerics';

export type ProjectionComponents = number | 'auto';
interface BaseProps { nComponents?: ProjectionComponents; eps?: number; computeInverseComponents?: boolean; randomState?: number; }

abstract class BaseRandomProjection extends TransformerBase {
    protected nComponents: ProjectionComponents; protected eps: number; protected computeInverseComponents: boolean; protected randomState?: number;
    protected componentsState: number[][] = []; protected inverseState: number[][] = []; protected nFeaturesState = 0;
    constructor(props: BaseProps = {}) {
        super(); const { nComponents = 'auto', eps = .1, computeInverseComponents = false, randomState } = props;
        if (nComponents !== 'auto' && (!Number.isInteger(nComponents) || nComponents < 1) || !Number.isFinite(eps) || eps <= 0 || eps >= 1) throw new Error('invalid random projection parameters');
        this.nComponents = nComponents; this.eps = eps; this.computeInverseComponents = computeInverseComponents; this.randomState = randomState;
    }
    public getParams(): Params { return { nComponents: this.nComponents, eps: this.eps, computeInverseComponents: this.computeInverseComponents, randomState: this.randomState }; }
    protected abstract generate(rows: number, columns: number, random: () => number): number[][];
    public fit(X: number[][]): void {
        this.nFeaturesState = validateMatrix(X); const target = this.nComponents === 'auto' ? Math.floor(4 * Math.log(X.length) / (this.eps ** 2 / 2 - this.eps ** 3 / 3)) : this.nComponents;
        if (target < 1) throw new Error(`eps=${this.eps} and nSamples=${X.length} imply an invalid target dimension of ${target}`);
        if (this.nComponents === 'auto' && target > this.nFeaturesState) throw new Error(`eps=${this.eps} and nSamples=${X.length} imply nComponents=${target}, which exceeds nFeatures=${this.nFeaturesState}`);
        this.componentsState = this.generate(target, this.nFeaturesState, createRandomGenerator(this.randomState));
        if (this.computeInverseComponents) {
            const gram = matMul(this.componentsState, transpose(this.componentsState));
            this.inverseState = matMul(transpose(this.componentsState), pseudoInverseSymmetric(gram));
        } else this.inverseState = [];
    }
    public transform(X: number[][]): number[][] {
        if (this.componentsState.length === 0) throw new Error('random projection is not fitted');
        if (X.some(row => row.length !== this.nFeaturesState || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match fitted feature count');
        return matMul(X, transpose(this.componentsState));
    }
    public inverseTransform(X: number[][]): number[][] {
        if (this.inverseState.length === 0) throw new Error('computeInverseComponents must be true before inverseTransform');
        if (X.some(row => row.length !== this.componentsState.length || row.some(value => !Number.isFinite(value)))) throw new Error('projected input shape differs from fitted projection');
        return matMul(X, transpose(this.inverseState));
    }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
}

export interface GaussianRandomProjectionProps extends BaseProps {}
export class GaussianRandomProjection extends BaseRandomProjection {
    protected generate(rows: number, columns: number, random: () => number): number[][] { return Array.from({ length: rows }, () => Array.from({ length: columns }, () => normalRandom(random) / Math.sqrt(rows))); }
}
registerEstimator('GaussianRandomProjection', GaussianRandomProjection);

export interface SparseRandomProjectionProps extends BaseProps { density?: number | 'auto'; denseOutput?: boolean; }
export class SparseRandomProjection extends BaseRandomProjection {
    private density: number | 'auto'; private denseOutput: boolean;
    constructor(props: SparseRandomProjectionProps = {}) { super(props); const { density = 'auto', denseOutput = false } = props; if (density !== 'auto' && (!Number.isFinite(density) || density <= 0 || density > 1)) throw new Error('density must be auto or in (0, 1]'); this.density = density; this.denseOutput = denseOutput; }
    public getParams(): Params { return { ...super.getParams(), density: this.density, denseOutput: this.denseOutput }; }
    protected generate(rows: number, columns: number, random: () => number): number[][] {
        const density = this.density === 'auto' ? 1 / Math.sqrt(columns) : this.density, scale = Math.sqrt(1 / density) / Math.sqrt(rows);
        return Array.from({ length: rows }, () => Array.from({ length: columns }, () => random() < density ? (random() < .5 ? -scale : scale) : 0));
    }
}
registerEstimator('SparseRandomProjection', SparseRandomProjection);
