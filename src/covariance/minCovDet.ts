import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';

export interface MinCovDetProps {
    supportFraction?: number;
    randomState?: number;
    assumeCentered?: boolean;
}

interface Candidate {
    indices: number[];
    location: number[];
    covariance: number[][];
    precision: number[][];
    logDet: number;
    rank: number;
}

function meanRows(X: number[][], centered: boolean): number[] {
    const mean = new Array(X[0].length).fill(0);
    if (centered) return mean;
    for (const row of X) for (let j = 0; j < row.length; j++) mean[j] += row[j];
    return mean.map(value => value / X.length);
}

function covarianceOf(X: number[][], location: number[]): number[][] {
    const p = location.length;
    const covariance = Array.from({ length: p }, () => new Array(p).fill(0));
    for (const row of X) for (let a = 0; a < p; a++) for (let b = 0; b <= a; b++) {
        covariance[a][b] += (row[a] - location[a]) * (row[b] - location[b]);
    }
    for (let a = 0; a < p; a++) for (let b = 0; b <= a; b++) {
        covariance[a][b] /= X.length;
        covariance[b][a] = covariance[a][b];
    }
    return covariance;
}

function pseudoInverseAndLogDet(A: number[][]): { inverse: number[][]; logDet: number; rank: number } {
    const n = A.length;
    const { values, vectors } = symmetricEigDecomposition(A);
    const scale = Math.max(0, ...values.map(Math.abs));
    const tolerance = Number.EPSILON * n * (scale || 1);
    const inverse = Array.from({ length: n }, () => new Array(n).fill(0));
    let logDet = 0;
    let rank = 0;
    for (let k = 0; k < values.length; k++) {
        if (!(values[k] > tolerance)) continue;
        rank++;
        logDet += Math.log(values[k]);
        const vector = vectors[k];
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            inverse[i][j] += vector[i] * vector[j] / values[k];
        }
    }
    return { inverse, logDet, rank };
}

function squaredMahalanobis(X: number[][], location: number[], precision: number[][]): number[] {
    return X.map(row => {
        const delta = row.map((value, j) => value - location[j]);
        let sum = 0;
        for (let i = 0; i < delta.length; i++) for (let j = 0; j < delta.length; j++) sum += delta[i] * precision[i][j] * delta[j];
        return Math.max(0, sum);
    });
}

function median(values: number[]): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const middle = (sorted.length - 1) / 2;
    return (sorted[Math.floor(middle)] + sorted[Math.ceil(middle)]) / 2;
}

function logGamma(z: number): number {
    const c = [0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    z--;
    let x = c[0];
    for (let i = 1; i < c.length; i++) x += c[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function gammaP(a: number, x: number): number {
    if (x <= 0) return 0;
    if (x < a + 1) {
        let sum = 1 / a, term = sum;
        for (let n = 1; n < 200; n++) {
            term *= x / (a + n); sum += term;
            if (Math.abs(term) < Math.abs(sum) * 1e-14) break;
        }
        return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }
    let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
    for (let i = 1; i < 200; i++) {
        const an = -i * (i - a); b += 2;
        d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d; const delta = d * c; h *= delta;
        if (Math.abs(delta - 1) < 1e-14) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

function chiSquareQuantile(probability: number, degrees: number): number {
    let low = 0, high = Math.max(1, degrees);
    while (gammaP(degrees / 2, high / 2) < probability) high *= 2;
    for (let i = 0; i < 100; i++) {
        const middle = (low + high) / 2;
        if (gammaP(degrees / 2, middle / 2) < probability) low = middle;
        else high = middle;
    }
    return (low + high) / 2;
}

function candidateFor(X: number[][], indices: number[], centered: boolean): Candidate | undefined {
    const subset = indices.map(i => X[i]);
    const location = meanRows(subset, centered);
    const covariance = covarianceOf(subset, location);
    const decomposition = pseudoInverseAndLogDet(covariance);
    if (decomposition.rank === 0) return undefined;
    return { indices, location, covariance, precision: decomposition.inverse, logDet: decomposition.logDet, rank: decomposition.rank };
}

function cStep(X: number[][], candidate: Candidate, h: number, centered: boolean): Candidate | undefined {
    const distances = squaredMahalanobis(X, candidate.location, candidate.precision);
    const indices = distances.map((distance, i) => ({ distance, i }))
        .sort((a, b) => a.distance - b.distance || a.i - b.i).slice(0, h).map(entry => entry.i);
    return candidateFor(X, indices, centered);
}

function compareCandidates(a: Candidate, b: Candidate): number {
    return b.rank - a.rank || a.logDet - b.logDet;
}

function elementalCandidates(
    X: number[][],
    h: number,
    p: number,
    random: () => number,
    starts: number,
): Candidate[] {
    const candidates: Candidate[] = [];
    for (let start = 0; start < starts; start++) {
        const indices = Array.from({ length: X.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        let candidate = candidateFor(X, indices.slice(0, p + 1), false);
        if (!candidate) continue;
        for (let step = 0; step < 2; step++) {
            candidate = cStep(X, candidate, h, false);
            if (!candidate) break;
        }
        if (candidate) candidates.push(candidate);
    }
    return candidates;
}

function advanceCandidates(candidates: Candidate[], X: number[][], h: number): Candidate[] {
    const advanced: Candidate[] = [];
    // Candidates produced on different subsets have incomparable determinants.
    // First evaluate every subset winner on the same target data, then rank.
    for (const initial of candidates) {
        let candidate: Candidate | undefined = initial;
        for (let step = 0; step < 2; step++) {
            candidate = cStep(X, candidate, h, false);
            if (!candidate) break;
        }
        if (candidate) advanced.push(candidate);
    }
    return advanced.sort(compareCandidates).slice(0, 10);
}

export class MinCovDet extends BaseEstimator {
    private supportFraction?: number;
    private randomState?: number;
    private assumeCentered: boolean;
    private locationState: number[] = [];
    private covarianceState: number[][] = [];
    private precisionState: number[][] = [];
    private supportState: boolean[] = [];
    private rawLocationState: number[] = [];
    private rawCovarianceState: number[][] = [];
    private rawSupportState: boolean[] = [];

    constructor(props: MinCovDetProps = {}) {
        super();
        const { supportFraction, randomState, assumeCentered = false } = props;
        if (supportFraction !== undefined && (!(supportFraction > 0) || supportFraction > 1)) throw new Error('supportFraction must be in (0, 1]');
        this.supportFraction = supportFraction;
        this.randomState = randomState;
        this.assumeCentered = assumeCentered;
    }

    public getParams(): Params { return { supportFraction: this.supportFraction, randomState: this.randomState, assumeCentered: this.assumeCentered }; }

    public fit(X: number[][]): void {
        if (X.length < 2 || X[0].length === 0 || X.some(row => row.length !== X[0].length)) throw new Error('X must be a rectangular matrix with at least two samples');
        const n = X.length, p = X[0].length;
        const h = this.supportFraction === undefined ? Math.ceil((n + p + 1) / 2) : Math.floor(this.supportFraction * n);
        if (h <= p) throw new Error('support set must contain more samples than features');
        const random = createRandomGenerator(this.randomState);
        let candidates: Candidate[];
        if (n <= 500) {
            candidates = elementalCandidates(X, h, p, random, Math.min(500, Math.max(100, n * 5)));
        } else {
            // FAST-MCD's large-sample path keeps expensive C-steps bounded:
            // build candidates on ~300-row subsets, optionally merge through
            // a 1500-row pool, and only send the best candidates to full X.
            const order = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(random() * (i + 1));
                [order[i], order[j]] = [order[j], order[i]];
            }
            candidates = [];
            const nSubsets = Math.max(1, Math.floor(n / 300));
            const startsPerSubset = Math.max(1, Math.floor(500 / nSubsets));
            for (let subsetIndex = 0; subsetIndex < nSubsets; subsetIndex++) {
                const start = Math.floor(subsetIndex * n / nSubsets);
                const end = Math.floor((subsetIndex + 1) * n / nSubsets);
                const subset = order.slice(start, end).map(i => X[i]);
                if (subset.length <= p) continue;
                const subsetH = Math.max(p + 1, Math.ceil(h * subset.length / n));
                candidates.push(...elementalCandidates(subset, subsetH, p, random, startsPerSubset).sort(compareCandidates).slice(0, 10));
            }
            if (n > 1500) {
                const pool = order.slice(0, 1500).map(i => X[i]);
                const poolH = Math.max(p + 1, Math.ceil(h * pool.length / n));
                candidates = advanceCandidates(candidates, pool, poolH);
            }
            candidates = advanceCandidates(candidates, X, h);
        }
        candidates.sort(compareCandidates);
        let best: Candidate | undefined;
        for (const initial of candidates.slice(0, 10)) {
            let candidate = initial;
            for (let iter = 0; iter < 100; iter++) {
                const next = cStep(X, candidate, h, false);
                if (!next) break;
                const converged = Math.abs(next.logDet - candidate.logDet) < 1e-9;
                candidate = next;
                if (converged) break;
            }
            if (!best || compareCandidates(candidate, best) < 0) best = candidate;
        }
        if (!best) throw new Error('FAST-MCD failed to find a nonsingular elemental subset');
        const raw = this.assumeCentered ? candidateFor(X, best.indices, true) : best;
        if (!raw) throw new Error('FAST-MCD raw covariance is singular');
        this.rawLocationState = raw.location;
        this.rawCovarianceState = raw.covariance;
        const rawIndexSet = new Set(best.indices);
        this.rawSupportState = X.map((_, i) => rawIndexSet.has(i));

        const rawDistances = squaredMahalanobis(X, raw.location, raw.precision);
        const correction = median(rawDistances) / chiSquareQuantile(0.5, p);
        if (!Number.isFinite(correction) || correction <= 0) throw new Error('FAST-MCD covariance correction failed');
        const correctedDistances = rawDistances.map(distance => distance / correction);
        const cutoff = chiSquareQuantile(0.975, p);
        let support = correctedDistances.map(distance => distance <= cutoff);
        if (support.filter(Boolean).length <= p) support = X.map((_, i) => best!.indices.includes(i));
        const reweighted = X.filter((_, i) => support[i]);
        this.locationState = meanRows(reweighted, this.assumeCentered);
        this.covarianceState = covarianceOf(reweighted, this.locationState);
        this.precisionState = pseudoInverseAndLogDet(this.covarianceState).inverse;
        this.supportState = support;
    }

    public mahalanobis(X: number[][]): number[] {
        if (this.precisionState.length === 0) throw new Error('MinCovDet is not fitted');
        return squaredMahalanobis(X, this.locationState, this.precisionState);
    }

    public get location(): number[] { return this.locationState.slice(); }
    public get covariance(): number[][] { return this.covarianceState.map(row => row.slice()); }
    public get precision(): number[][] { return this.precisionState.map(row => row.slice()); }
    public get support(): boolean[] { return this.supportState.slice(); }
    public get rawLocation(): number[] { return this.rawLocationState.slice(); }
    public get rawCovariance(): number[][] { return this.rawCovarianceState.map(row => row.slice()); }
    public get rawSupport(): boolean[] { return this.rawSupportState.slice(); }
}
registerEstimator('MinCovDet', MinCovDet);
