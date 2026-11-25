import { ClusterBase } from '../base/cluster';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import { EMeansTraining, nearestCenter, resolveEMeansRecipe, squaredDistance } from './eMeansTraining';

export interface EMeansProps {
    nClusters?: number;
    epochs?: number;
    batchSize?: number;
    randomState?: number;
    /** Fraction of mean moving cluster mass. Zero disables revival. */
    revivalThreshold?: number;
    /** Optional initial centers, one row per cluster. */
    initCenters?: number[][];
}

function positiveInteger(value: number, name: string) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError('EMeans: ' + name + ' must be a positive integer');
}

function validateRows(rows: number[][], width: number) {
    for (const row of rows) {
        if (!Array.isArray(row) || row.length !== width) throw new Error('EMeans: inconsistent feature dimension');
        for (const x of row) if (!Number.isFinite(x)) throw new Error('EMeans: expected finite feature values');
    }
}

function shufflePrefix(order: Uint32Array, size: number, random: () => number) {
    for (let i = 0; i < size; i++) {
        const j = i + Math.floor(random() * (order.length - i));
        const value = order[i];
        order[i] = order[j];
        order[j] = value;
    }
}

/** Mini-batch e-means with exact squared Euclidean assignment. */
export class EMeans extends ClusterBase {
    private readonly options: EMeansProps & { nClusters: number; epochs: number };
    private centers: number[][] | null = null;
    private labels: number[] = [];
    private inertia = Infinity;

    constructor({ nClusters = 8, epochs = 5, batchSize, randomState, revivalThreshold, initCenters }: EMeansProps = {}) {
        super();
        positiveInteger(nClusters, 'nClusters');
        positiveInteger(epochs, 'epochs');
        if (batchSize !== undefined) positiveInteger(batchSize, 'batchSize');
        if (randomState !== undefined && !Number.isSafeInteger(randomState)) throw new RangeError('EMeans: expected integer randomState');
        if (revivalThreshold !== undefined && (!Number.isFinite(revivalThreshold) || revivalThreshold < 0)) {
            throw new RangeError('EMeans: revivalThreshold must be finite and nonnegative');
        }
        if (initCenters !== undefined) {
            if (initCenters.length !== nClusters || !initCenters[0]?.length) throw new Error('EMeans: expected one initial center per cluster');
            validateRows(initCenters, initCenters[0].length);
        }
        this.options = { nClusters, epochs, batchSize, randomState, revivalThreshold, initCenters: initCenters?.map(row => row.slice()) };
    }

    getParams(): Params {
        return { ...this.options, initCenters: this.options.initCenters?.map(row => row.slice()) };
    }

    fit(X: number[][], sampleWeights?: number[]): this {
        const { nClusters: k, epochs, batchSize, randomState, revivalThreshold, initCenters } = this.options;
        const n = X.length, d = X[0]?.length;
        if (n < k || !d) throw new Error('EMeans: expected at least nClusters rows with one or more features');
        validateRows(X, d);
        if (sampleWeights !== undefined) {
            if (sampleWeights.length !== n) throw new Error('EMeans: expected one unit weight per row');
            for (const weight of sampleWeights) {
                if (weight !== 1) throw new Error('EMeans: only unit sample weights are supported');
            }
        }
        if (initCenters && initCenters[0].length !== d) throw new Error('EMeans: initial center feature dimension differs from data');
        // Map signed seeds into the nonnegative input range of the shared generator.
        const seed = randomState === undefined ? undefined : ((randomState % 2147483647) + 2147483647) % 2147483647;
        const random = createRandomGenerator(seed);
        const order = Uint32Array.from({ length: n }, (_, i) => i);
        if (!initCenters && k < n) shufflePrefix(order, k, random);
        const initial = initCenters ?? Array.from(order.subarray(0, k), i => X[i]);
        const recipe = resolveEMeansRecipe(n, d, k, epochs, batchSize);
        const training = new EMeansTraining(initial.map(row => row.slice()), recipe, revivalThreshold ?? (d <= 2 ? 0 : 0.1), random);
        for (let epoch = 0; epoch < epochs; epoch++) {
            shufflePrefix(order, n, random);
            for (let start = 0; start < n; start += recipe.batchSize) {
                training.update(X, order, start, Math.min(start + recipe.batchSize, n), epoch === epochs - 1);
            }
        }
        const centers = training.finalize();
        const labels = X.map(row => nearestCenter(row, centers));
        let inertia = 0;
        for (let i = 0; i < n; i++) inertia += squaredDistance(X[i], centers[labels[i]]);
        if (!Number.isFinite(inertia) || centers.some(row => row.some(x => !Number.isFinite(x)))) {
            throw new RangeError('EMeans: training exceeds numeric range');
        }
        this.centers = centers;
        this.labels = labels;
        this.inertia = inertia;
        return this;
    }

    fitPredict(X: number[][], sampleWeights?: number[]): number[] { return this.fit(X, sampleWeights).getLabels(); }
    predict(X: number[][]): number[] {
        if (!this.centers) throw new Error('EMeans: call fit before predict');
        validateRows(X, this.centers[0].length);
        return X.map(row => nearestCenter(row, this.centers));
    }
    getCentroids(): number[][] | null { return this.centers?.map(row => row.slice()) ?? null; }
    getLabels(): number[] { return this.labels.slice(); }
    getInertia(): number { return this.inertia; }
}

registerEstimator('EMeans', EMeans);
