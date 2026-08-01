import { BaseEstimator } from '../base';

interface PredictiveEstimator extends BaseEstimator { predict(X: number[][]): number[]; score?(X: number[][], y: number[]): number; }
export interface PermutationImportanceProps { nRepeats?: number; randomState?: number; scoring?: (estimator: PredictiveEstimator, X: number[][], y: number[]) => number; }
export interface PermutationImportanceResult { importancesMean: number[]; importancesStd: number[]; importances: number[][]; }

/** NumPy RandomState-compatible MT19937 stream for sklearn seeded permutation parity. */
function mt19937(seed?: number): { uint32: () => number; interval: (max: number) => number } {
    if (seed === undefined) return { uint32: () => Math.floor(Math.random() * 0x100000000), interval(max) { return Math.floor(Math.random() * (max + 1)); } };
    const state = new Uint32Array(624); state[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) { const previous = state[i - 1] ^ (state[i - 1] >>> 30); state[i] = (Math.imul(1812433253, previous) + i) >>> 0; }
    let index = 624;
    const uint32 = () => {
        if (index >= 624) { for (let i = 0; i < 624; i++) { const y = (state[i] & 0x80000000) | (state[(i + 1) % 624] & 0x7fffffff); state[i] = state[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0); } index = 0; }
        let y = state[index++]; y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18; return y >>> 0;
    };
    return { uint32, interval(max: number) { let mask = max; mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16; let value: number; do value = uint32() & mask; while (value > max); return value; } };
}

/** Model-agnostic feature importance measured by repeated score degradation after shuffling. */
export function permutationImportance(estimator: PredictiveEstimator, X: number[][], y: number[], props: PermutationImportanceProps = {}): PermutationImportanceResult {
    const { nRepeats = 5, randomState, scoring } = props;
    if (!Number.isInteger(nRepeats) || nRepeats <= 0 || X.length === 0 || X.length !== y.length || X.some(row => row.length !== X[0].length)) throw new Error('invalid permutation importance data or nRepeats');
    const score = scoring ?? ((model, features, target) => {
        if (typeof model.score !== 'function') throw new Error('estimator must implement score or a scoring callback must be provided');
        return model.score(features, target);
    });
    const baseline = score(estimator, X, y);
    const seedSource = mt19937(randomState), permutationSeed = seedSource.interval(0x7fffffff);
    const importances = Array.from({ length: X[0].length }, () => new Array(nRepeats).fill(0));
    for (let feature = 0; feature < X[0].length; feature++) {
        const random = mt19937(permutationSeed);
        const order = Array.from({ length: X.length }, (_, i) => i);
        const shuffled = X.map(row => row.slice());
        for (let repeat = 0; repeat < nRepeats; repeat++) {
        for (let i = order.length - 1; i > 0; i--) { const j = random.interval(i); [order[i], order[j]] = [order[j], order[i]]; }
        const column = order.map(index => shuffled[index][feature]);
        for (let i = 0; i < X.length; i++) shuffled[i][feature] = column[i];
        importances[feature][repeat] = baseline - score(estimator, shuffled, y);
        }
    }
    const importancesMean = importances.map(row => row.reduce((sum, value) => sum + value, 0) / row.length);
    const importancesStd = importances.map((row, i) => Math.sqrt(row.reduce((sum, value) => sum + (value - importancesMean[i]) ** 2, 0) / row.length));
    return { importancesMean, importancesStd, importances };
}

export interface PartialDependenceProps { gridResolution?: number; percentiles?: [number, number]; }
export interface PartialDependenceResult { gridValues: number[][]; average: number[][] | number[][][]; }

function quantile(values: number[], q: number): number {
    const sorted = values.slice().sort((a, b) => a - b), position = Math.max(0, Math.min(sorted.length - 1, sorted.length * q + .4 + .2 * q - 1));
    const lo = Math.floor(position), hi = Math.ceil(position), fraction = position - lo;
    return sorted[lo] * (1 - fraction) + sorted[hi] * fraction;
}

/** Brute-force partial dependence for one or two numeric features. */
export function partialDependence(estimator: Pick<PredictiveEstimator, 'predict'>, X: number[][], features: number[], props: PartialDependenceProps = {}): PartialDependenceResult {
    const { gridResolution = 100, percentiles = [.05, .95] } = props;
    if (X.length === 0 || X.some(row => row.length !== X[0].length) || features.length < 1 || features.length > 2 || new Set(features).size !== features.length) throw new Error('partialDependence supports one or two valid features');
    if (!Number.isInteger(gridResolution) || gridResolution < 2 || percentiles[0] < 0 || percentiles[1] > 1 || percentiles[0] >= percentiles[1]) throw new Error('invalid partial dependence grid options');
    const gridValues = features.map(feature => {
        if (!Number.isInteger(feature) || feature < 0 || feature >= X[0].length) throw new Error('feature index is out of bounds');
        const values = X.map(row => row[feature]);
        const unique = Array.from(new Set(values)).sort((a, b) => a - b);
        if (unique.length < gridResolution) return unique;
        const lo = quantile(values, percentiles[0]), hi = quantile(values, percentiles[1]);
        return Array.from({ length: gridResolution }, (_, i) => lo + i * (hi - lo) / (gridResolution - 1));
    });
    const combinations: number[][] = features.length === 1 ? gridValues[0].map(v => [v]) : gridValues[0].flatMap(a => gridValues[1].map(b => [a, b]));
    const flattened = combinations.map(values => {
        const modified = X.map(row => { const copy = row.slice(); features.forEach((feature, i) => copy[feature] = values[i]); return copy; });
        const predictions = estimator.predict(modified);
        return predictions.reduce((sum, value) => sum + value, 0) / predictions.length;
    });
    const average: number[][] | number[][][] = features.length === 1
        ? [flattened]
        : [gridValues[0].map((_, i) => flattened.slice(i * gridValues[1].length, (i + 1) * gridValues[1].length))];
    return { gridValues, average };
}
