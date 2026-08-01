import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { Distance } from '../metrics';
import { BallTree } from './ballTree';
import { KDTree } from './kdTree';

export type NeighborsAlgorithm = 'auto' | 'ballTree' | 'kdTree' | 'brute';

export interface NearestNeighborsProps {
    nNeighbors?: number;
    radius?: number;
    algorithm?: NeighborsAlgorithm;
    leafSize?: number;
    metric?: Distance.IDistanceType;
    p?: number;
}

/** Unsupervised nearest-neighbor search over the existing KDTree/BallTree primitives. */
export class NearestNeighbors extends BaseEstimator {
    private nNeighbors: number;
    private radius: number;
    private algorithm: NeighborsAlgorithm;
    private leafSize: number;
    private metric: Distance.IDistanceType;
    private p: number;
    private trainingState: number[][] = [];
    private nFeaturesState = 0;

    constructor(props: NearestNeighborsProps = {}) {
        super();
        const { nNeighbors = 5, radius = 1, algorithm = 'auto', leafSize = 30, metric = 'minkowski', p = 2 } = props;
        if (!Number.isInteger(nNeighbors) || nNeighbors <= 0) throw new Error('nNeighbors must be a positive integer');
        if (!Number.isFinite(radius) || radius < 0) throw new Error('radius must be non-negative');
        if (!['auto', 'ballTree', 'kdTree', 'brute'].includes(algorithm)) throw new Error('invalid neighbors algorithm');
        if (!Number.isInteger(leafSize) || leafSize <= 0) throw new Error('leafSize must be a positive integer');
        if (!Number.isFinite(p) || p < 1) throw new Error('p must be at least 1 for a valid Minkowski metric');
        Distance.useDistance(metric);
        this.nNeighbors = nNeighbors; this.radius = radius; this.algorithm = algorithm;
        this.leafSize = leafSize; this.metric = metric; this.p = p;
    }

    public getParams(): Params {
        return { nNeighbors: this.nNeighbors, radius: this.radius, algorithm: this.algorithm, leafSize: this.leafSize, metric: this.metric, p: this.p };
    }

    public fit(X: number[][]): void {
        if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(v => !Number.isFinite(v)))) {
            throw new Error('X must be a non-empty finite rectangular matrix');
        }
        this.trainingState = X.map(row => row.slice());
        this.nFeaturesState = X[0].length;
    }

    private validateQueries(X: number[][]): void {
        if (this.trainingState.length === 0) throw new Error('NearestNeighbors is not fitted');
        if (X.some(row => row.length !== this.nFeaturesState || row.some(v => !Number.isFinite(v)))) throw new Error('query dimensionality does not match fitted data');
    }

    private resolvedAlgorithm(): Exclude<NeighborsAlgorithm, 'auto'> {
        if (this.algorithm !== 'auto') return this.algorithm;
        return this.nFeaturesState > 15 ? 'brute' : 'kdTree';
    }

    private query(X: number[][], k: number): { distances: number[][]; indices: number[][] } {
        const algorithm = this.resolvedAlgorithm();
        if (algorithm === 'kdTree' || algorithm === 'ballTree') {
            const result = algorithm === 'kdTree'
                ? new KDTree(this.trainingState, this.leafSize, this.metric, this.p).query(X, k)
                : new BallTree(this.trainingState, this.leafSize, this.metric, this.p).query(X, k);
            for (let i = 0; i < result.indices.length; i++) {
                const pairs = result.indices[i].map((index, j) => ({ index, distance: result.distances[i][j] }))
                    .sort((a, b) => a.distance - b.distance || a.index - b.index);
                result.indices[i] = pairs.map(item => item.index); result.distances[i] = pairs.map(item => item.distance);
            }
            return result;
        }
        const distance = Distance.useDistance(this.metric);
        const pairs = X.map(row => this.trainingState.map((sample, index) => ({ index, distance: distance(row, sample, this.p) }))
            .sort((a, b) => a.distance - b.distance || a.index - b.index).slice(0, k));
        return { distances: pairs.map(row => row.map(item => item.distance)), indices: pairs.map(row => row.map(item => item.index)) };
    }

    public kneighbors(X?: number[][], nNeighbors = this.nNeighbors, returnDistance = true): number[][] | { distances: number[][]; indices: number[][] } {
        if (!Number.isInteger(nNeighbors) || nNeighbors <= 0) throw new Error('nNeighbors must be a positive integer');
        const ownQuery = X === undefined;
        const queries = ownQuery ? this.trainingState : X;
        this.validateQueries(queries);
        if (nNeighbors > this.trainingState.length - (ownQuery ? 1 : 0)) throw new Error('nNeighbors exceeds the number of fitted samples');
        const result = this.query(queries, nNeighbors + (ownQuery ? 1 : 0));
        if (ownQuery) for (let i = 0; i < result.indices.length; i++) {
            const position = result.indices[i].indexOf(i);
            if (position >= 0) { result.indices[i].splice(position, 1); result.distances[i].splice(position, 1); }
            result.indices[i] = result.indices[i].slice(0, nNeighbors);
            result.distances[i] = result.distances[i].slice(0, nNeighbors);
        }
        return returnDistance ? result : result.indices;
    }

    public radiusNeighbors(X?: number[][], radius = this.radius, returnDistance = true, sortResults = false): number[][] | { distances: number[][]; indices: number[][] } {
        if (!Number.isFinite(radius) || radius < 0) throw new Error('radius must be non-negative');
        if (sortResults && !returnDistance) throw new Error('sortResults requires returnDistance=true');
        const ownQuery = X === undefined;
        const queries = ownQuery ? this.trainingState : X;
        this.validateQueries(queries);
        const algorithm = this.resolvedAlgorithm();
        let result: { distances: number[][]; indices: number[][] };
        const distance = Distance.useDistance(this.metric);
        if (algorithm === 'kdTree' || algorithm === 'ballTree') {
            const tree = algorithm === 'kdTree'
                ? new KDTree(this.trainingState, this.leafSize, this.metric, this.p)
                : new BallTree(this.trainingState, this.leafSize, this.metric, this.p);
            const indices = tree.queryRadius(queries, radius, false) as number[][];
            result = { indices, distances: indices.map((row, i) => row.map(index => distance(queries[i], this.trainingState[index], this.p))) };
        } else {
            const pairs = queries.map(row => this.trainingState.map((sample, index) => ({ index, distance: distance(row, sample, this.p) }))
                .filter(item => item.distance <= radius));
            result = { distances: pairs.map(row => row.map(item => item.distance)), indices: pairs.map(row => row.map(item => item.index)) };
        }
        if (ownQuery) for (let i = 0; i < result.indices.length; i++) {
            const position = result.indices[i].indexOf(i);
            if (position >= 0) { result.indices[i].splice(position, 1); result.distances[i].splice(position, 1); }
        }
        if (sortResults) for (let i = 0; i < result.indices.length; i++) {
            const pairs = result.indices[i].map((index, j) => ({ index, distance: result.distances[i][j] })).sort((a, b) => a.distance - b.distance || a.index - b.index);
            result.indices[i] = pairs.map(item => item.index); result.distances[i] = pairs.map(item => item.distance);
        }
        return returnDistance ? result : result.indices;
    }
}

registerEstimator('NearestNeighbors', NearestNeighbors);
