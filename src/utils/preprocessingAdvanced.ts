import { BaseEstimator, TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from './random';

function validateNumericMatrix(X: number[][]): number {
    if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(v => !Number.isFinite(v)))) throw new Error('X must be a non-empty finite rectangular matrix');
    return X[0].length;
}
function quantileSorted(values: number[], q: number): number {
    const position = (values.length - 1) * q, lo = Math.floor(position), hi = Math.ceil(position), fraction = position - lo;
    return values[lo] * (1 - fraction) + values[hi] * fraction;
}
function compareLabels(a: string | number | boolean | null, b: string | number | boolean | null): number {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    const left = String(a), right = String(b); return left < right ? -1 : left > right ? 1 : 0;
}

export type SplineKnots = 'uniform' | 'quantile' | number[][];
export type SplineExtrapolation = 'error' | 'constant' | 'linear' | 'continue' | 'periodic';
export interface SplineTransformerProps { nKnots?: number; degree?: number; knots?: SplineKnots; extrapolation?: SplineExtrapolation; includeBias?: boolean; }

/** Univariate B-spline basis expansion applied independently to every numeric feature. */
export class SplineTransformer extends TransformerBase {
    private nKnots: number; private degree: number; private knots: SplineKnots;
    private extrapolation: SplineExtrapolation; private includeBias: boolean;
    private knotVectorsState: number[][] = []; private boundariesState: Array<[number, number]> = []; private nFeaturesState = 0;
    constructor(props: SplineTransformerProps = {}) {
        super();
        const { nKnots = 5, degree = 3, knots = 'uniform', extrapolation = 'constant', includeBias = true } = props;
        if (!Number.isInteger(nKnots) || nKnots < 2 || !Number.isInteger(degree) || degree < 0) throw new Error('nKnots must be >= 2 and degree must be a non-negative integer');
        if (!['uniform', 'quantile'].includes(knots as string) && !Array.isArray(knots)) throw new Error('invalid knots');
        if (!['error', 'constant', 'linear', 'continue', 'periodic'].includes(extrapolation)) throw new Error('invalid extrapolation');
        if (extrapolation === 'periodic' && !Array.isArray(knots) && nKnots <= degree) throw new Error('periodic splines require more knots than degree');
        this.nKnots = nKnots; this.degree = degree; this.knots = Array.isArray(knots) ? knots.map(row => row.slice()) : knots;
        this.extrapolation = extrapolation; this.includeBias = includeBias;
    }
    public getParams(): Params { return { nKnots: this.nKnots, degree: this.degree, knots: Array.isArray(this.knots) ? this.knots.map(row => row.slice()) : this.knots, extrapolation: this.extrapolation, includeBias: this.includeBias }; }
    public fit(X: number[][]): void {
        this.nFeaturesState = validateNumericMatrix(X);
        const bases: number[][] = [];
        for (let j = 0; j < this.nFeaturesState; j++) {
            const values = X.map(row => row[j]).sort((a, b) => a - b);
            let base: number[];
            if (Array.isArray(this.knots)) {
                if (this.knots.length < 2 || this.knots.some(row => row.length !== this.nFeaturesState)) throw new Error('explicit knots must have shape [nKnots][nFeatures]');
                base = this.knots.map(row => row[j]);
            } else if (this.knots === 'quantile') base = Array.from({ length: this.nKnots }, (_, i) => quantileSorted(values, i / (this.nKnots - 1)));
            else base = Array.from({ length: this.nKnots }, (_, i) => values[0] + i * (values[values.length - 1] - values[0]) / (this.nKnots - 1));
            if (base.some((value, i) => i > 0 && value <= base[i - 1])) throw new Error('knots must be strictly increasing; constant features are unsupported');
            if (this.extrapolation === 'periodic' && base.length <= this.degree) throw new Error('periodic splines require more knots than degree');
            bases.push(base);
        }
        this.boundariesState = bases.map(base => [base[0], base[base.length - 1]]);
        this.knotVectorsState = bases.map(base => {
            if (this.extrapolation === 'periodic') {
                const period = base[base.length - 1] - base[0];
                return [
                    ...base.slice(-(this.degree + 1), -1).map(value => value - period),
                    ...base,
                    ...base.slice(1, this.degree + 1).map(value => value + period),
                ];
            }
            const leftStep = base[1] - base[0], rightStep = base[base.length - 1] - base[base.length - 2];
            const left = Array.from({ length: this.degree }, (_, i) => base[0] - (this.degree - i) * leftStep);
            const right = Array.from({ length: this.degree }, (_, i) => base[base.length - 1] + (i + 1) * rightStep);
            return [...left, ...base, ...right];
        });
    }
    private basisAtDegree(x: number, knots: number[], degree: number): number[] {
        const count = knots.length - degree - 1;
        let span: number;
        if (x <= knots[degree]) span = degree;
        else if (x >= knots[count]) span = count - 1;
        else {
            let lo = degree, hi = count;
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x < knots[mid]) hi = mid; else lo = mid; }
            span = lo;
        }
        const local = new Array(degree + 1).fill(0), left = new Array(degree + 1).fill(0), right = new Array(degree + 1).fill(0); local[0] = 1;
        for (let order = 1; order <= degree; order++) {
            left[order] = x - knots[span + 1 - order]; right[order] = knots[span + order] - x; let saved = 0;
            for (let r = 0; r < order; r++) { const denominator = right[r + 1] + left[order - r], term = denominator === 0 ? 0 : local[r] / denominator; local[r] = saved + right[r + 1] * term; saved = left[order - r] * term; }
            local[order] = saved;
        }
        const output = new Array(count).fill(0); for (let i = 0; i <= degree; i++) output[span - degree + i] = local[i]; return output;
    }
    private basis(x: number, knots: number[]): number[] { return this.basisAtDegree(x, knots, this.degree); }
    private basisDerivative(x: number, knots: number[]): number[] {
        if (this.degree === 0) return new Array(knots.length - 1).fill(0);
        const lower = this.basisAtDegree(x, knots, this.degree - 1), count = knots.length - this.degree - 1;
        return Array.from({ length: count }, (_, i) => {
            const leftDenominator = knots[i + this.degree] - knots[i], rightDenominator = knots[i + this.degree + 1] - knots[i + 1];
            return (leftDenominator === 0 ? 0 : this.degree * lower[i] / leftDenominator) - (rightDenominator === 0 ? 0 : this.degree * lower[i + 1] / rightDenominator);
        });
    }
    private featureBasis(value: number, feature: number): number[] {
        const [lo, hi] = this.boundariesState[feature], knots = this.knotVectorsState[feature];
        if (this.extrapolation === 'error' && (value < lo || value > hi)) throw new Error('X contains values outside the fitted spline range');
        if (this.extrapolation === 'periodic') {
            const period = hi - lo; value = lo + ((((value - lo) % period) + period) % period);
            const raw = this.basis(value, knots), count = raw.length - this.degree, output = raw.slice(0, count);
            for (let i = 0; i < this.degree; i++) output[i] += raw[count + i];
            return output;
        }
        if (this.extrapolation === 'constant') value = Math.min(hi, Math.max(lo, value));
        if (this.extrapolation === 'linear' && (value < lo || value > hi)) {
            const boundary = value < lo ? lo : hi, distance = value - boundary;
            const center = this.basis(boundary, knots), derivative = this.basisDerivative(boundary, knots);
            return center.map((base, i) => base + distance * derivative[i]);
        }
        return this.basis(value, knots);
    }
    public transform(X: number[][]): number[][] {
        if (this.knotVectorsState.length === 0) throw new Error('SplineTransformer is not fitted');
        if (validateNumericMatrix(X) !== this.nFeaturesState) throw new Error('feature count does not match fitted data');
        return X.map(row => row.flatMap((value, feature) => {
            const basis = this.featureBasis(value, feature);
            return this.includeBias ? basis : basis.slice(0, -1);
        }));
    }
}
registerEstimator('SplineTransformer', SplineTransformer);

export type Category = string | number | boolean | null;
export type TargetLabel = string | number | boolean;
export interface TargetEncoderProps { categories?: 'auto' | Category[][]; targetType?: 'auto' | 'continuous' | 'binary' | 'multiclass'; smooth?: 'auto' | number; cv?: number; shuffle?: boolean; randomState?: number; }
interface EncodingStats { global: number[]; maps: Array<Map<Category, number[]>>; categories: Category[][]; targetType: 'continuous' | 'binary' | 'multiclass'; targetClasses: TargetLabel[]; }

/** Supervised categorical encoder with smoothed target means and leakage-safe cross-fitted fitTransform. */
export class TargetEncoder extends BaseEstimator {
    private categories: 'auto' | Category[][]; private targetType: 'auto' | 'continuous' | 'binary' | 'multiclass'; private smooth: 'auto' | number;
    private cv: number; private shuffle: boolean; private randomState?: number; private statsState?: EncodingStats;
    constructor(props: TargetEncoderProps = {}) {
        super();
        const { categories = 'auto', targetType = 'auto', smooth = 'auto', cv = 5, shuffle = true, randomState } = props;
        if (smooth !== 'auto' && (!Number.isFinite(smooth) || smooth < 0)) throw new Error('smooth must be auto or non-negative');
        if (!Number.isInteger(cv) || cv < 2) throw new Error('cv must be at least 2');
        this.categories = Array.isArray(categories) ? categories.map(row => row.slice()) : categories;
        this.targetType = targetType; this.smooth = smooth; this.cv = cv; this.shuffle = shuffle; this.randomState = randomState;
    }
    public getParams(): Params { return { categories: Array.isArray(this.categories) ? this.categories.map(row => row.slice()) : this.categories, targetType: this.targetType, smooth: this.smooth, cv: this.cv, shuffle: this.shuffle, randomState: this.randomState }; }
    private validate(X: Category[][], y?: TargetLabel[]): number {
        if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length)) throw new Error('X must be a non-empty rectangular categorical matrix');
        if (y && (y.length !== X.length || y.some(value => typeof value === 'number' && !Number.isFinite(value)))) throw new Error('X and y must be aligned and finite');
        return X[0].length;
    }
    private prepareTargets(y: TargetLabel[]): { values: number[][]; type: 'continuous' | 'binary' | 'multiclass'; classes: TargetLabel[] } {
        const unique = Array.from(new Set(y)).sort(compareLabels);
        let type: 'continuous' | 'binary' | 'multiclass';
        if (this.targetType !== 'auto') type = this.targetType;
        else if (unique.length <= 2) type = 'binary';
        else if (y.every(value => typeof value === 'number') && (y as number[]).some(value => !Number.isInteger(value))) type = 'continuous';
        else type = 'multiclass';
        if (type === 'continuous') {
            if (!y.every(value => typeof value === 'number')) throw new Error('continuous targets must be numeric');
            return { values: (y as number[]).map(value => [value]), type, classes: [] };
        }
        if (type === 'binary') {
            if (unique.length < 1 || unique.length > 2) throw new Error('binary targetType requires one or two target classes');
            return { values: y.map(value => [value === unique[1] ? 1 : 0]), type, classes: unique };
        }
        if (unique.length < 2) throw new Error('multiclass targetType requires at least two classes');
        return { values: y.map(value => unique.map(label => value === label ? 1 : 0)), type, classes: unique };
    }
    private compute(X: Category[][], targets: number[][], type: 'continuous' | 'binary' | 'multiclass', targetClasses: TargetLabel[], fixedCategories?: Category[][]): EncodingStats {
        const nFeatures = this.validate(X), nOutputs = targets[0].length;
        const global = Array.from({ length: nOutputs }, (_, output) => targets.reduce((sum, row) => sum + row[output], 0) / targets.length);
        const globalVariance = global.map((mean, output) => targets.reduce((sum, row) => sum + (row[output] - mean) ** 2, 0) / targets.length);
        const categories = fixedCategories ?? (Array.isArray(this.categories) ? this.categories.map(row => row.slice()) : Array.from({ length: nFeatures }, (_, j) => Array.from(new Set(X.map(row => row[j]))).sort(compareLabels)));
        if (categories.length !== nFeatures) throw new Error('categories must contain one array per feature');
        const maps = categories.map((allowed, j) => {
            const map = new Map<Category, number[]>();
            for (const category of allowed) {
                const selected = targets.filter((_, i) => X[i][j] === category);
                if (selected.length === 0) { map.set(category, global.slice()); continue; }
                map.set(category, Array.from({ length: nOutputs }, (_, output) => {
                    const mean = selected.reduce((sum, row) => sum + row[output], 0) / selected.length;
                    if (this.smooth === 'auto') {
                        const variance = selected.reduce((sum, row) => sum + (row[output] - mean) ** 2, 0) / selected.length;
                        const weight = globalVariance[output] === 0 ? 1 : globalVariance[output] * selected.length / (globalVariance[output] * selected.length + variance);
                        return weight * mean + (1 - weight) * global[output];
                    }
                    return (selected.reduce((sum, row) => sum + row[output], 0) + this.smooth * global[output]) / (selected.length + this.smooth || 1);
                }));
            }
            return map;
        });
        return { global, maps, categories: categories.map(row => row.slice()), targetType: type, targetClasses: targetClasses.slice() };
    }
    public fit(X: Category[][], y: TargetLabel[]): void { this.validate(X, y); const prepared = this.prepareTargets(y); this.statsState = this.compute(X, prepared.values, prepared.type, prepared.classes); }
    private encode(X: Category[][], stats: EncodingStats): number[][] {
        if (this.validate(X) !== stats.maps.length) throw new Error('feature count does not match fitted data');
        return X.map(row => row.flatMap((category, j) => stats.maps[j].get(category) ?? stats.global));
    }
    public transform(X: Category[][]): number[][] { if (!this.statsState) throw new Error('TargetEncoder is not fitted'); return this.encode(X, this.statsState); }
    public fitTransform(X: Category[][], y: TargetLabel[]): number[][] {
        this.validate(X, y);
        if (this.cv > X.length) throw new Error('cv cannot exceed the number of samples');
        const prepared = this.prepareTargets(y), full = this.compute(X, prepared.values, prepared.type, prepared.classes), random = createRandomGenerator(this.randomState);
        const shuffle = (indices: number[]) => { if (this.shuffle) for (let i = indices.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [indices[i], indices[j]] = [indices[j], indices[i]]; } return indices; };
        const folds: number[][] = Array.from({ length: this.cv }, () => []);
        if (prepared.type === 'continuous') {
            const order = shuffle(Array.from({ length: X.length }, (_, i) => i)), base = Math.floor(order.length / this.cv), remainder = order.length % this.cv; let offset = 0;
            for (let fold = 0; fold < this.cv; fold++) { const size = base + (fold < remainder ? 1 : 0); folds[fold] = order.slice(offset, offset + size); offset += size; }
        } else {
            const foldClasses = Array.from(new Set(y)), encoded = y.map(value => foldClasses.findIndex(label => label === value)), sorted = encoded.slice().sort((a, b) => a - b);
            const allocation = Array.from({ length: this.cv }, (_, fold) => Array.from({ length: foldClasses.length }, (_, label) => sorted.filter((value, position) => position % this.cv === fold && value === label).length));
            for (let label = 0; label < foldClasses.length; label++) {
                const group = encoded.flatMap((value, index) => value === label ? [index] : []), assignments = allocation.flatMap((row, fold) => new Array(row[label]).fill(fold));
                shuffle(assignments);
                group.forEach((sample, index) => folds[assignments[index]].push(sample));
            }
        }
        const result = Array.from({ length: X.length }, () => new Array(X[0].length * prepared.values[0].length).fill(0));
        for (let fold = 0; fold < this.cv; fold++) {
            const validation = folds[fold], held = new Set(validation), training = Array.from({ length: X.length }, (_, i) => i).filter(index => !held.has(index));
            const stats = this.compute(training.map(index => X[index]), training.map(index => prepared.values[index]), prepared.type, prepared.classes, full.categories);
            const encoded = this.encode(validation.map(index => X[index]), stats);
            validation.forEach((index, i) => result[index] = encoded[i]);
        }
        this.statsState = full;
        return result;
    }
}
registerEstimator('TargetEncoder', TargetEncoder);
