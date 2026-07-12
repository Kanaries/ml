import { ClusterBase } from '../base/cluster';
import { registerEstimator, Params } from '../base/estimator';
import { kmeansPlusPlus } from './kmeans_plusplus';
import { createRandomGenerator } from '../utils/random';

/**
 * MiniBatchKMeans, following sklearn.cluster.MiniBatchKMeans (the Sculley
 * 2010 web-scale k-means algorithm as refined by sklearn):
 *
 * 1. `nInit` candidate initializations (k-means++ or random pick) are each
 *    computed on a random subsample of size init_size = min(3 * batchSize,
 *    nSamples) (at least 3 * nClusters), evaluated by one mini-batch step on
 *    a fixed validation subsample; the candidate with the lowest inertia
 *    wins.
 * 2. Main loop of (maxIter * nSamples / batchSize) steps. Each step draws
 *    `batchSize` samples uniformly at random *with replacement* (sklearn's
 *    `random_state.randint`), assigns them to the nearest centers, and moves
 *    each center with the count-based learning rate:
 *        c_new = (c_old * count + sum(batch members)) / (count + members)
 *    which is the batched form of Sculley's per-sample c += (x - c)/count.
 * 3. Centers whose accumulated weight falls below
 *    reassignmentRatio * max(weights) are periodically (every ~10*nClusters
 *    processed samples) reassigned to random samples from the current batch.
 * 4. Early stopping (sklearn's _mini_batch_convergence): stop when the
 *    normalized squared center change is <= tol (variance-scaled, only when
 *    tol > 0), or when the exponentially weighted average of the batch
 *    inertia has not improved for `maxNoImprovement` consecutive steps.
 * 5. Final labels and inertia are computed on the full dataset.
 */

export interface MiniBatchKMeansProps {
    /** number of clusters */
    nClusters?: number;
    /** initialization strategy */
    initParams?: 'k-means++' | 'random';
    /** maximum number of passes (epoch-equivalents) over the data */
    maxIter?: number;
    /** mini-batch size (clamped to the number of samples) */
    batchSize?: number;
    /**
     * relative tolerance on the center change for convergence, scaled by
     * the mean feature variance; 0 (default) disables this criterion
     */
    tol?: number;
    /**
     * stop when the smoothed batch inertia has not improved for this many
     * consecutive steps; null/0 disables early stopping on inertia
     */
    maxNoImprovement?: number | null;
    /** number of random initializations evaluated */
    nInit?: number;
    /**
     * centers with accumulated weight below this fraction of the maximum
     * center weight are periodically reassigned; 0 disables reassignment
     */
    reassignmentRatio?: number;
    /** seed for reproducible sampling and initialization */
    randomState?: number;
}

interface MiniBatchStepResult {
    /** sum of squared distances of the batch to its nearest (old) centers */
    inertia: number;
    /** total squared movement of the centers in this step */
    squaredDiff: number;
}

export class MiniBatchKMeans extends ClusterBase {
    private nClusters: number;
    private initParams: 'k-means++' | 'random';
    private maxIter: number;
    private batchSize: number;
    private tol: number;
    private maxNoImprovement: number | null;
    private nInit: number;
    private reassignmentRatio: number;
    private randomState?: number;
    private centers: number[][] | null = null;
    private counts: number[] = [];
    private labels: number[] = [];
    private inertia: number = Infinity;

    constructor(props: MiniBatchKMeansProps = {}) {
        super();
        const {
            nClusters = 8,
            initParams = 'k-means++',
            maxIter = 100,
            batchSize = 1024,
            tol = 0,
            maxNoImprovement = 10,
            nInit = 3,
            reassignmentRatio = 0.01,
            randomState,
        } = props;
        if (initParams !== 'k-means++' && initParams !== 'random') {
            throw new Error(`Unknown initParams "${initParams}"; expected 'k-means++' or 'random'`);
        }
        if (batchSize < 1) {
            throw new Error('batchSize must be at least 1');
        }
        this.nClusters = nClusters;
        this.initParams = initParams;
        this.maxIter = maxIter;
        this.batchSize = batchSize;
        this.tol = tol;
        this.maxNoImprovement = maxNoImprovement;
        this.nInit = nInit;
        this.reassignmentRatio = reassignmentRatio;
        this.randomState = randomState;
    }

    public getParams(): Params {
        return {
            nClusters: this.nClusters,
            initParams: this.initParams,
            maxIter: this.maxIter,
            batchSize: this.batchSize,
            tol: this.tol,
            maxNoImprovement: this.maxNoImprovement,
            nInit: this.nInit,
            reassignmentRatio: this.reassignmentRatio,
            randomState: this.randomState,
        };
    }

    public fit(samplesX: number[][], sampleWeights?: number[]): this {
        this.centers = null;
        this.counts = [];
        this.labels = [];
        this.inertia = Infinity;
        const n = samplesX.length;
        if (n === 0) return this;
        const weights = sampleWeights ? sampleWeights : samplesX.map(() => 1);
        const rng = createRandomGenerator(this.randomState);
        const k = this.nClusters;
        const batchSize = Math.min(this.batchSize, n);

        // ------------------------------------------------------------------
        // initialization: nInit candidates, each judged by one mini-batch
        // step on a fixed validation subsample (sklearn's fit)
        // ------------------------------------------------------------------
        let initSize = 3 * batchSize;
        if (initSize < 3 * k) initSize = 3 * k;
        if (initSize > n) initSize = n;
        const validationIndices: number[] = new Array(initSize);
        for (let i = 0; i < initSize; i++) validationIndices[i] = Math.floor(rng() * n);
        let bestCenters: number[][] | null = null;
        let bestCounts: number[] | null = null;
        let bestInertia = Infinity;
        const runs = Math.max(1, this.nInit);
        for (let run = 0; run < runs; run++) {
            const centers = this.initCenters(samplesX, weights, initSize, rng);
            const counts: number[] = new Array(k).fill(0);
            const { inertia } = this.miniBatchStep(samplesX, weights, validationIndices, centers, counts, false, rng);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCenters = centers;
                bestCounts = counts;
            }
        }
        let centers = bestCenters as number[][];
        const counts = bestCounts as number[];

        // ------------------------------------------------------------------
        // main mini-batch loop
        // ------------------------------------------------------------------
        const nSteps = Math.floor((this.maxIter * n) / batchSize);
        // variance-scaled absolute tolerance, like sklearn's _tolerance
        const tolAbs = this.tol > 0 ? this.tol * meanFeatureVariance(samplesX) : 0;
        // EWA smoothing factor over the batch inertia (sklearn)
        let alpha = (batchSize * 2) / (n + 1);
        if (alpha > 1) alpha = 1;
        let ewaInertia: number | null = null;
        let ewaInertiaMin: number | null = null;
        let noImprovement = 0;
        let samplesSinceReassign = 0;

        for (let step = 0; step < nSteps; step++) {
            const batchIndices: number[] = new Array(batchSize);
            for (let i = 0; i < batchSize; i++) batchIndices[i] = Math.floor(rng() * n);

            // periodic low-count center reassignment, every ~10 * nClusters
            // processed samples (sklearn's _n_since_last_reassign logic)
            samplesSinceReassign += batchSize;
            let randomReassign = false;
            if (samplesSinceReassign >= 10 * k) {
                randomReassign = true;
                samplesSinceReassign = 0;
            }

            const { inertia, squaredDiff } = this.miniBatchStep(
                samplesX, weights, batchIndices, centers, counts, randomReassign, rng
            );

            // convergence checks (sklearn's _mini_batch_convergence)
            let batchWeight = 0;
            for (const idx of batchIndices) batchWeight += weights[idx];
            const batchInertia = batchWeight > 0 ? inertia / batchWeight : 0;
            if (this.tol > 0 && squaredDiff <= tolAbs) {
                break;
            }
            if (ewaInertia === null) {
                ewaInertia = batchInertia;
            } else {
                ewaInertia = ewaInertia * (1 - alpha) + batchInertia * alpha;
            }
            if (ewaInertiaMin === null || ewaInertia < ewaInertiaMin) {
                ewaInertiaMin = ewaInertia;
                noImprovement = 0;
            } else {
                noImprovement += 1;
            }
            if (this.maxNoImprovement !== null && this.maxNoImprovement > 0
                && noImprovement >= this.maxNoImprovement) {
                break;
            }
        }

        // final full-dataset assignment
        const { labels, inertia } = assignToCenters(samplesX, centers, weights);
        this.centers = centers;
        this.counts = counts;
        this.labels = labels;
        this.inertia = inertia;
        return this;
    }

    public fitPredict(samplesX: number[][], sampleWeights?: number[]): number[] {
        this.fit(samplesX, sampleWeights);
        return this.labels;
    }

    /** Assign each (new) sample to its nearest fitted center. */
    public predict(samplesX: number[][]): number[] {
        if (this.centers === null) {
            throw new Error('MiniBatchKMeans: call fit before predict');
        }
        return assignToCenters(samplesX, this.centers, samplesX.map(() => 1)).labels;
    }

    public getCentroids(): number[][] | null {
        return this.centers;
    }

    public getInertia(): number {
        return this.inertia;
    }

    public getLabels(): number[] {
        return this.labels;
    }

    /** One candidate initialization on a random subsample of size initSize. */
    private initCenters(
        samplesX: number[][],
        weights: number[],
        initSize: number,
        rng: () => number
    ): number[][] {
        const n = samplesX.length;
        const k = this.nClusters;
        // random subsample without replacement (partial Fisher-Yates over
        // the index range, sklearn uses a permutation prefix)
        const indices = sampleWithoutReplacement(n, initSize, rng);
        const subX = indices.map((i) => samplesX[i]);
        const subW = indices.map((i) => weights[i]);
        if (this.initParams === 'k-means++') {
            return kmeansPlusPlus(subX, k, subW, rng).centers.map((c) => c.slice());
        }
        // 'random': k distinct samples from the subsample (repeats only when
        // the subsample is smaller than k)
        const pick = sampleWithoutReplacement(subX.length, Math.min(k, subX.length), rng);
        const centers = pick.map((i) => subX[i].slice());
        while (centers.length < k) {
            centers.push(subX[Math.floor(rng() * subX.length)].slice());
        }
        return centers;
    }

    /**
     * One mini-batch update: assign the batch to the nearest centers
     * (returning the pre-update inertia), then move each touched center with
     * its count-based learning rate. Optionally random-reassign low-weight
     * centers to samples from this batch. Mutates `centers` and `counts`.
     */
    private miniBatchStep(
        samplesX: number[][],
        weights: number[],
        batchIndices: number[],
        centers: number[][],
        counts: number[],
        randomReassign: boolean,
        rng: () => number
    ): MiniBatchStepResult {
        const k = centers.length;
        const dim = samplesX[0].length;
        let inertia = 0;
        const wsum: number[] = new Array(k).fill(0);
        const acc: number[][] = new Array(k);
        for (let c = 0; c < k; c++) acc[c] = new Array(dim).fill(0);

        for (const idx of batchIndices) {
            const x = samplesX[idx];
            const w = weights[idx];
            let best = 0;
            let bestDist = Infinity;
            for (let c = 0; c < k; c++) {
                let d = 0;
                for (let j = 0; j < dim; j++) {
                    const diff = centers[c][j] - x[j];
                    d += diff * diff;
                }
                if (d < bestDist) {
                    bestDist = d;
                    best = c;
                }
            }
            inertia += bestDist * w;
            wsum[best] += w;
            for (let j = 0; j < dim; j++) acc[best][j] += x[j] * w;
        }

        let squaredDiff = 0;
        for (let c = 0; c < k; c++) {
            if (wsum[c] <= 0) continue;
            const newCount = counts[c] + wsum[c];
            for (let j = 0; j < dim; j++) {
                const updated = (centers[c][j] * counts[c] + acc[c][j]) / newCount;
                const diff = updated - centers[c][j];
                squaredDiff += diff * diff;
                centers[c][j] = updated;
            }
            counts[c] = newCount;
        }

        if (randomReassign && this.reassignmentRatio > 0) {
            let maxCount = 0;
            for (let c = 0; c < k; c++) maxCount = Math.max(maxCount, counts[c]);
            let toReassign: number[] = [];
            for (let c = 0; c < k; c++) {
                if (counts[c] < this.reassignmentRatio * maxCount) toReassign.push(c);
            }
            // never reassign more than half the batch worth of centers
            const cap = Math.floor(0.5 * batchIndices.length);
            if (toReassign.length > cap) {
                toReassign = toReassign
                    .slice()
                    .sort((a, b) => counts[a] - counts[b])
                    .slice(0, cap);
            }
            if (toReassign.length > 0) {
                const picks = sampleWithoutReplacement(
                    batchIndices.length, Math.min(toReassign.length, batchIndices.length), rng
                );
                const reassignSet = new Set(toReassign);
                // reset counts to the smallest surviving count so the new
                // center is not immediately reassigned again (sklearn)
                let minSurviving = Infinity;
                for (let c = 0; c < k; c++) {
                    if (!reassignSet.has(c)) minSurviving = Math.min(minSurviving, counts[c]);
                }
                if (!Number.isFinite(minSurviving)) minSurviving = 1;
                for (let r = 0; r < toReassign.length; r++) {
                    const c = toReassign[r];
                    const sample = samplesX[batchIndices[picks[r % picks.length]]];
                    centers[c] = sample.slice();
                    counts[c] = minSurviving;
                }
            }
        }
        return { inertia, squaredDiff };
    }
}
registerEstimator('MiniBatchKMeans', MiniBatchKMeans);

/** Labels + weighted inertia of X under fixed centers. */
function assignToCenters(
    samplesX: number[][],
    centers: number[][],
    weights: number[]
): { labels: number[]; inertia: number } {
    const labels: number[] = new Array(samplesX.length).fill(0);
    let inertia = 0;
    for (let i = 0; i < samplesX.length; i++) {
        const x = samplesX[i];
        let best = 0;
        let bestDist = Infinity;
        for (let c = 0; c < centers.length; c++) {
            let d = 0;
            for (let j = 0; j < x.length; j++) {
                const diff = centers[c][j] - x[j];
                d += diff * diff;
            }
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        labels[i] = best;
        inertia += bestDist * weights[i];
    }
    return { labels, inertia };
}

/** `size` distinct indices from [0, n) via partial Fisher-Yates. */
function sampleWithoutReplacement(n: number, size: number, rng: () => number): number[] {
    const pool: number[] = new Array(n);
    for (let i = 0; i < n; i++) pool[i] = i;
    const m = Math.min(size, n);
    for (let i = 0; i < m; i++) {
        const j = i + Math.floor(rng() * (n - i));
        const tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
    }
    return pool.slice(0, m);
}

/** Mean per-feature variance of X (sklearn's _tolerance scaling). */
function meanFeatureVariance(samplesX: number[][]): number {
    const n = samplesX.length;
    if (n === 0) return 0;
    const dim = samplesX[0].length;
    let total = 0;
    for (let j = 0; j < dim; j++) {
        let mean = 0;
        for (let i = 0; i < n; i++) mean += samplesX[i][j];
        mean /= n;
        let varSum = 0;
        for (let i = 0; i < n; i++) {
            const diff = samplesX[i][j] - mean;
            varSum += diff * diff;
        }
        total += varSum / n;
    }
    return total / dim;
}
