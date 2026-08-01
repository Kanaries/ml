import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { symmetricEigen } from '../algebra/eigen';

export interface IsomapProps {
    nNeighbors?: number;
    nComponents?: number;
}

function euclidean(a: number[], b: number[]): number {
    let total = 0;
    for (let j = 0; j < a.length; j++) total += (a[j] - b[j]) ** 2;
    return Math.sqrt(total);
}

function connectedComponents(graph: number[][]): number[][] {
    const seen = new Array(graph.length).fill(false);
    const components: number[][] = [];
    for (let root = 0; root < graph.length; root++) {
        if (seen[root]) continue;
        const component: number[] = [];
        const stack = [root];
        seen[root] = true;
        while (stack.length > 0) {
            const node = stack.pop()!;
            component.push(node);
            for (let next = 0; next < graph.length; next++) {
                if (!seen[next] && Number.isFinite(graph[node][next])) {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        components.push(component);
    }
    return components;
}

function connectComponents(graph: number[][], X: number[][]): number {
    const components = connectedComponents(graph);
    for (let a = 0; a < components.length; a++) for (let b = 0; b < a; b++) {
        let bestI = -1, bestJ = -1, bestDistance = Infinity;
        for (const i of components[a]) for (const j of components[b]) {
            const d = euclidean(X[i], X[j]);
            if (d < bestDistance) { bestDistance = d; bestI = i; bestJ = j; }
        }
        graph[bestI][bestJ] = bestDistance;
        graph[bestJ][bestI] = bestDistance;
    }
    return components.length;
}

export class Isomap extends TransformerBase<number[][], number[][]> {
    private nNeighbors: number;
    private nComponents: number;
    private trainX: number[][] = [];
    private geodesic: number[][] = [];
    private eigenvalues: number[] = [];
    private eigenvectors: number[][] = [];
    private squaredColumnMeans: number[] = [];
    private squaredTotalMean = 0;
    private embeddingState: number[][] = [];

    constructor(props: IsomapProps = {}) {
        super();
        const { nNeighbors = 5, nComponents = 2 } = props;
        if (!Number.isInteger(nNeighbors) || nNeighbors < 1) throw new Error('nNeighbors must be a positive integer');
        if (!Number.isInteger(nComponents) || nComponents < 1) throw new Error('nComponents must be a positive integer');
        this.nNeighbors = nNeighbors;
        this.nComponents = nComponents;
    }

    public getParams(): Params { return { nNeighbors: this.nNeighbors, nComponents: this.nComponents }; }

    public fit(X: number[][]): void {
        if (X.length < 2 || X[0].length === 0 || X.some(row => row.length !== X[0].length)) throw new Error('X must contain at least two rectangular samples');
        this.trainX = X.map(row => row.slice());
        const n = X.length;
        if (this.nComponents > n) throw new Error('nComponents cannot exceed the number of samples');
        const graph = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 0 : Infinity));
        for (let i = 0; i < n; i++) {
            const nearest = X.map((row, j) => ({ j, d: i === j ? Infinity : euclidean(X[i], row) }))
                .sort((a, b) => a.d - b.d || a.j - b.j)
                .slice(0, Math.min(this.nNeighbors, n - 1));
            for (const { j, d } of nearest) {
                graph[i][j] = Math.min(graph[i][j], d);
                graph[j][i] = Math.min(graph[j][i], d);
            }
        }
        const componentCount = connectComponents(graph, X);
        if (componentCount > 1) {
            // Matches sklearn's non-precomputed Isomap recovery: bridge each
            // pair of connected components at their closest Euclidean points.
            // eslint-disable-next-line no-console
            console.warn(`[@kanaries/ml] Isomap neighbor graph has ${componentCount} connected components; adding minimum-distance bridges.`);
        }
        for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const through = graph[i][k] + graph[k][j];
            if (through < graph[i][j]) graph[i][j] = through;
        }
        this.geodesic = graph;
        const squared = graph.map(row => row.map(value => value * value));
        const rowMeans = squared.map(row => row.reduce((a, b) => a + b, 0) / n);
        this.squaredColumnMeans = Array.from({ length: n }, (_, j) => squared.reduce((sum, row) => sum + row[j], 0) / n);
        this.squaredTotalMean = rowMeans.reduce((a, b) => a + b, 0) / n;
        const kernel = squared.map((row, i) => row.map((value, j) => -0.5 * (value - rowMeans[i] - this.squaredColumnMeans[j] + this.squaredTotalMean)));
        const eigen = symmetricEigen(kernel, this.nComponents, { seed: 0, maxIter: 2000 });
        this.eigenvalues = eigen.values.map(value => Math.max(0, value));
        this.eigenvectors = eigen.vectors;
        this.embeddingState = Array.from({ length: n }, (_, i) => this.eigenvectors.map((vector, c) => vector[i] * Math.sqrt(this.eigenvalues[c])));
    }

    public transform(X: number[][]): number[][] {
        if (this.trainX.length === 0) throw new Error('Isomap is not fitted');
        return X.map(sample => {
            const neighbors = this.trainX.map((row, i) => ({ i, d: euclidean(sample, row) })).sort((a, b) => a.d - b.d || a.i - b.i).slice(0, Math.min(this.nNeighbors, this.trainX.length));
            const distances = this.trainX.map((_, j) => {
                let shortest = Infinity;
                for (const hit of neighbors) shortest = Math.min(shortest, hit.d + this.geodesic[hit.i][j]);
                return shortest;
            });
            const squared = distances.map(value => value * value);
            const mean = squared.reduce((a, b) => a + b, 0) / squared.length;
            const centered = squared.map((value, j) => -0.5 * (value - mean - this.squaredColumnMeans[j] + this.squaredTotalMean));
            return this.eigenvectors.map((vector, c) => this.eigenvalues[c] <= 1e-12
                ? 0
                : vector.reduce((sum, value, j) => sum + value * centered[j], 0) / Math.sqrt(this.eigenvalues[c]));
        });
    }

    public fitTransform(X: number[][]): number[][] { this.fit(X); return this.embeddingState.map(row => row.slice()); }
    public get embedding(): number[][] { return this.embeddingState.map(row => row.slice()); }
    public get distMatrix(): number[][] { return this.geodesic.map(row => row.slice()); }
}
registerEstimator('Isomap', Isomap);
