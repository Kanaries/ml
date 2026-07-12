/**
 * Additional preprocessing transformers (sklearn-matching semantics, camelCase
 * props). Everything here follows the estimator contract
 * (docs/estimator-contract.md): props-object constructors, `getParams()`,
 * `registerEstimator()` next to each class, and JSON-serializable state.
 *
 * Numerical notes:
 *  - Quantiles use linear interpolation, matching `numpy.percentile`'s default.
 *  - The inverse standard-normal CDF is Peter John Acklam's rational
 *    approximation (relative error < 1.15e-9); the normal CDF uses the
 *    Abramowitz & Stegun 7.1.26 erf approximation (abs error < 1.5e-7).
 *  - `PowerTransformer` estimates lambda per feature by maximum likelihood
 *    with a coarse grid scan over [-5, 5] followed by golden-section
 *    refinement (sklearn uses scipy's Brent on a (-2, 2) bracket; both find
 *    the same unimodal optimum to ~1e-9 on well-behaved data).
 */
import { BaseEstimator, TransformerBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { createRandomGenerator } from './random';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function validateMatrix(X: number[][]): void {
    if (X.length === 0) {
        throw new Error('X must be non-empty');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must contain at least one feature');
    }
    for (let i = 1; i < X.length; i++) {
        if (X[i].length !== nFeatures) {
            throw new Error('X must be a rectangular matrix');
        }
    }
}

function assertSameFeatureCount(X: number[][], featureCount: number, message: string): void {
    validateMatrix(X);
    if (X[0].length !== featureCount) {
        throw new Error(message);
    }
}

/** Non-NaN values of column `j`. */
function observedColumn(X: number[][], j: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < X.length; i++) {
        if (!Number.isNaN(X[i][j])) {
            out.push(X[i][j]);
        }
    }
    return out;
}

/**
 * Linear-interpolation quantile of an ascending-sorted array, `q` in
 * [0, 100] (numpy.percentile's default 'linear' method).
 */
function quantileSorted(sorted: number[], q: number): number {
    const h = (sorted.length - 1) * (q / 100);
    const lo = Math.floor(h);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

/** np.interp semantics: piecewise-linear with clamping outside the range. */
function interp1d(x: number, xp: number[], fp: number[]): number {
    const last = xp.length - 1;
    if (x >= xp[last]) return fp[last];
    if (x < xp[0]) return fp[0];
    // binary search: largest lo with xp[lo] <= x
    let lo = 0;
    let hi = last;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xp[mid] <= x) lo = mid;
        else hi = mid;
    }
    if (x === xp[lo] || xp[hi] === xp[lo]) return fp[lo];
    const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
}

/**
 * Inverse standard-normal CDF (Peter John Acklam's approximation,
 * relative error < 1.15e-9 over the full domain).
 */
export function normPpf(p: number): number {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
        1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
        6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
        -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pLow = 0.02425;
    if (p < pLow) {
        const q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - pLow) {
        const q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Standard-normal CDF via the Abramowitz & Stegun 7.1.26 erf approximation. */
export function normCdf(x: number): number {
    const z = x / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * Math.abs(z));
    const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const erfAbs = 1 - poly * Math.exp(-z * z);
    const erf = z >= 0 ? erfAbs : -erfAbs;
    return 0.5 * (1 + erf);
}

function meanOf(values: number[]): number {
    let sum = 0;
    for (const v of values) sum += v;
    return sum / values.length;
}

/** Population (ddof=0) standard deviation, like sklearn's StandardScaler. */
function populationStd(values: number[], mean: number): number {
    let sq = 0;
    for (const v of values) {
        const d = v - mean;
        sq += d * d;
    }
    return Math.sqrt(sq / values.length);
}

/**
 * 1-D scalar minimizer: coarse grid scan over [lo, hi] to bracket the global
 * minimum, then golden-section refinement of the winning cell.
 */
function minimizeScalar(f: (x: number) => number, lo: number, hi: number): number {
    const GRID = 100;
    let bestX = lo;
    let bestF = Infinity;
    for (let i = 0; i <= GRID; i++) {
        const x = lo + (i * (hi - lo)) / GRID;
        const fx = f(x);
        if (fx < bestF) {
            bestF = fx;
            bestX = x;
        }
    }
    const step = (hi - lo) / GRID;
    let a = Math.max(lo, bestX - step);
    let b = Math.min(hi, bestX + step);
    const invPhi = (Math.sqrt(5) - 1) / 2;
    let c = b - invPhi * (b - a);
    let d = a + invPhi * (b - a);
    let fc = f(c);
    let fd = f(d);
    for (let iter = 0; iter < 200 && b - a > 1e-12; iter++) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = b - invPhi * (b - a);
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + invPhi * (b - a);
            fd = f(d);
        }
    }
    return (a + b) / 2;
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

export interface RobustScalerProps {
    withCentering?: boolean;
    withScaling?: boolean;
    /** Percentile pair (qMin, qMax), 0 <= qMin < qMax <= 100. Default IQR. */
    quantileRange?: [number, number];
    /** Scale so that normally-distributed features get unit variance. */
    unitVariance?: boolean;
}

/**
 * Scale features using statistics that are robust to outliers: remove the
 * median and scale by the quantile range (IQR by default). NaN values are
 * ignored when computing the fit statistics and pass through `transform`.
 */
export class RobustScaler extends TransformerBase {
    private withCentering: boolean;
    private withScaling: boolean;
    private quantileRange: [number, number];
    private unitVariance: boolean;
    private centers: number[];
    private scales: number[];
    private fitted: boolean;

    constructor(props: RobustScalerProps = {}) {
        super();
        const { withCentering = true, withScaling = true, quantileRange = [25, 75], unitVariance = false } = props;
        const [qMin, qMax] = quantileRange;
        if (!(qMin >= 0 && qMax <= 100 && qMin < qMax)) {
            throw new Error('quantileRange must satisfy 0 <= qMin < qMax <= 100');
        }
        this.withCentering = withCentering;
        this.withScaling = withScaling;
        this.quantileRange = [qMin, qMax];
        this.unitVariance = unitVariance;
        this.centers = [];
        this.scales = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            withCentering: this.withCentering,
            withScaling: this.withScaling,
            quantileRange: this.quantileRange,
            unitVariance: this.unitVariance,
        };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        const [qMin, qMax] = this.quantileRange;
        this.centers = new Array(nFeatures).fill(0);
        this.scales = new Array(nFeatures).fill(1);
        for (let j = 0; j < nFeatures; j++) {
            const observed = observedColumn(X, j);
            if (observed.length === 0) {
                throw new Error(`Feature ${j} has no observed values`);
            }
            const sorted = observed.slice().sort((a, b) => a - b);
            if (this.withCentering) {
                this.centers[j] = quantileSorted(sorted, 50);
            }
            if (this.withScaling) {
                let scale = quantileSorted(sorted, qMax) - quantileSorted(sorted, qMin);
                if (scale === 0) {
                    scale = 1; // sklearn's _handle_zeros_in_scale
                }
                if (this.unitVariance) {
                    scale /= normPpf(qMax / 100) - normPpf(qMin / 100);
                }
                this.scales[j] = scale;
            }
        }
        this.fitted = true;
    }

    private assertFittedAndShape(X: number[][], method: string): void {
        if (!this.fitted) {
            throw new Error(`RobustScaler must be fitted before calling ${method}`);
        }
        assertSameFeatureCount(X, this.centers.length, 'X has different number of features than fitted data');
    }

    public transform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'transform');
        return X.map(row => row.map((value, j) => {
            let ans = value;
            if (this.withCentering) ans -= this.centers[j];
            if (this.withScaling) ans /= this.scales[j];
            return ans;
        }));
    }

    public inverseTransform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'inverseTransform');
        return X.map(row => row.map((value, j) => {
            let ans = value;
            if (this.withScaling) ans *= this.scales[j];
            if (this.withCentering) ans += this.centers[j];
            return ans;
        }));
    }
}
registerEstimator('RobustScaler', RobustScaler);

// ---------------------------------------------------------------------------
// PowerTransformer
// ---------------------------------------------------------------------------

export interface PowerTransformerProps {
    method?: 'yeo-johnson' | 'box-cox';
    /** Apply zero-mean, unit-variance normalization after the power transform. */
    standardize?: boolean;
}

const LAMBDA_EPS = 1e-12;

function yeoJohnson(x: number, lambda: number): number {
    if (Number.isNaN(x)) return NaN;
    if (x >= 0) {
        return Math.abs(lambda) < LAMBDA_EPS ? Math.log1p(x) : (Math.pow(x + 1, lambda) - 1) / lambda;
    }
    return Math.abs(lambda - 2) < LAMBDA_EPS
        ? -Math.log1p(-x)
        : -(Math.pow(1 - x, 2 - lambda) - 1) / (2 - lambda);
}

function yeoJohnsonInverse(y: number, lambda: number): number {
    if (Number.isNaN(y)) return NaN;
    if (y >= 0) {
        return Math.abs(lambda) < LAMBDA_EPS ? Math.expm1(y) : Math.pow(y * lambda + 1, 1 / lambda) - 1;
    }
    return Math.abs(lambda - 2) < LAMBDA_EPS
        ? -Math.expm1(-y)
        : 1 - Math.pow(1 - (2 - lambda) * y, 1 / (2 - lambda));
}

function boxCox(x: number, lambda: number): number {
    if (Number.isNaN(x)) return NaN;
    return Math.abs(lambda) < LAMBDA_EPS ? Math.log(x) : (Math.pow(x, lambda) - 1) / lambda;
}

function boxCoxInverse(y: number, lambda: number): number {
    if (Number.isNaN(y)) return NaN;
    return Math.abs(lambda) < LAMBDA_EPS ? Math.exp(y) : Math.pow(y * lambda + 1, 1 / lambda);
}

/**
 * Apply a power transform featurewise to make the data more Gaussian-like.
 * Lambda is estimated per feature by maximizing the (Box-Cox or Yeo-Johnson)
 * log-likelihood with a grid scan over [-5, 5] plus golden-section
 * refinement. NaN values are ignored during fit and pass through transform.
 */
export class PowerTransformer extends TransformerBase {
    private method: 'yeo-johnson' | 'box-cox';
    private standardize: boolean;
    private lambdas: number[];
    private means: number[];
    private scales: number[];
    private fitted: boolean;

    constructor(props: PowerTransformerProps = {}) {
        super();
        const { method = 'yeo-johnson', standardize = true } = props;
        if (method !== 'yeo-johnson' && method !== 'box-cox') {
            throw new Error(`Unknown method "${method}". Expected 'yeo-johnson' or 'box-cox'.`);
        }
        this.method = method;
        this.standardize = standardize;
        this.lambdas = [];
        this.means = [];
        this.scales = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { method: this.method, standardize: this.standardize };
    }

    private applyPower(x: number, lambda: number): number {
        return this.method === 'box-cox' ? boxCox(x, lambda) : yeoJohnson(x, lambda);
    }

    private applyPowerInverse(y: number, lambda: number): number {
        return this.method === 'box-cox' ? boxCoxInverse(y, lambda) : yeoJohnsonInverse(y, lambda);
    }

    /** Negative log-likelihood of `lambda` for the observed column values. */
    private negativeLogLikelihood(observed: number[], lambda: number): number {
        const n = observed.length;
        const transformed = observed.map(x => this.applyPower(x, lambda));
        const mean = meanOf(transformed);
        let variance = 0;
        for (const t of transformed) {
            const d = t - mean;
            variance += d * d;
        }
        variance /= n;
        if (!Number.isFinite(variance) || variance <= 0) {
            return Infinity;
        }
        let jacobian = 0;
        if (this.method === 'box-cox') {
            for (const x of observed) jacobian += Math.log(x);
        } else {
            for (const x of observed) jacobian += Math.sign(x) * Math.log1p(Math.abs(x));
        }
        // llf = -n/2 * ln(var) + (lambda - 1) * jacobian; minimize the negation
        return (n / 2) * Math.log(variance) - (lambda - 1) * jacobian;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.lambdas = new Array(nFeatures).fill(1);
        this.means = new Array(nFeatures).fill(0);
        this.scales = new Array(nFeatures).fill(1);
        for (let j = 0; j < nFeatures; j++) {
            const observed = observedColumn(X, j);
            if (observed.length === 0) {
                throw new Error(`Feature ${j} has no observed values`);
            }
            if (this.method === 'box-cox' && observed.some(v => v <= 0)) {
                throw new Error('The Box-Cox transformation can only be applied to strictly positive data');
            }
            const lambda = minimizeScalar(l => this.negativeLogLikelihood(observed, l), -5, 5);
            this.lambdas[j] = lambda;
            if (this.standardize) {
                const transformed = observed.map(x => this.applyPower(x, lambda));
                const mean = meanOf(transformed);
                const std = populationStd(transformed, mean);
                this.means[j] = mean;
                this.scales[j] = std === 0 ? 1 : std;
            }
        }
        this.fitted = true;
    }

    private assertFittedAndShape(X: number[][], method: string): void {
        if (!this.fitted) {
            throw new Error(`PowerTransformer must be fitted before calling ${method}`);
        }
        assertSameFeatureCount(X, this.lambdas.length, 'X has different number of features than fitted data');
    }

    public transform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'transform');
        return X.map(row => row.map((value, j) => {
            let ans = this.applyPower(value, this.lambdas[j]);
            if (this.standardize) {
                ans = (ans - this.means[j]) / this.scales[j];
            }
            return ans;
        }));
    }

    public inverseTransform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'inverseTransform');
        return X.map(row => row.map((value, j) => {
            let ans = value;
            if (this.standardize) {
                ans = ans * this.scales[j] + this.means[j];
            }
            return this.applyPowerInverse(ans, this.lambdas[j]);
        }));
    }
}
registerEstimator('PowerTransformer', PowerTransformer);

// ---------------------------------------------------------------------------
// QuantileTransformer
// ---------------------------------------------------------------------------

export interface QuantileTransformerProps {
    nQuantiles?: number;
    outputDistribution?: 'uniform' | 'normal';
    /** Maximum number of samples used to estimate the quantiles. */
    subsample?: number;
    randomState?: number;
}

/** sklearn's BOUNDS_THRESHOLD: normal outputs are clipped at ppf(1e-7). */
const BOUNDS_THRESHOLD = 1e-7;

/** Partial Fisher-Yates: `k` distinct indices out of `n`. */
function sampleWithoutReplacement(n: number, k: number, rand: () => number): number[] {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
        const j = i + Math.floor(rand() * (n - i));
        const t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
    return idx.slice(0, k);
}

/**
 * Transform features to follow a uniform or normal distribution using
 * quantile information. The effective number of quantiles is silently
 * clamped to the number of samples (sklearn warns; we clamp silently).
 * NaN values are ignored during fit and pass through transform.
 */
export class QuantileTransformer extends TransformerBase {
    private nQuantiles: number;
    private outputDistribution: 'uniform' | 'normal';
    private subsample: number;
    private randomState?: number;
    private references: number[];
    private quantiles: number[][];
    private fitted: boolean;

    constructor(props: QuantileTransformerProps = {}) {
        super();
        const { nQuantiles = 1000, outputDistribution = 'uniform', subsample = 100000, randomState } = props;
        if (!Number.isInteger(nQuantiles) || nQuantiles < 1) {
            throw new Error('nQuantiles must be a positive integer');
        }
        if (!Number.isInteger(subsample) || subsample < 1) {
            throw new Error('subsample must be a positive integer');
        }
        if (outputDistribution !== 'uniform' && outputDistribution !== 'normal') {
            throw new Error(`Unknown outputDistribution "${outputDistribution}". Expected 'uniform' or 'normal'.`);
        }
        this.nQuantiles = nQuantiles;
        this.outputDistribution = outputDistribution;
        this.subsample = subsample;
        this.randomState = randomState;
        this.references = [];
        this.quantiles = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            nQuantiles: this.nQuantiles,
            outputDistribution: this.outputDistribution,
            subsample: this.subsample,
            randomState: this.randomState,
        };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nSamples = X.length;
        const nFeatures = X[0].length;
        const effective = Math.max(1, Math.min(this.nQuantiles, nSamples));
        this.references = Array.from({ length: effective }, (_, i) => (effective === 1 ? 0 : i / (effective - 1)));
        this.quantiles = [];
        const rand = createRandomGenerator(this.randomState);
        for (let j = 0; j < nFeatures; j++) {
            let column: number[];
            if (nSamples > this.subsample) {
                const indices = sampleWithoutReplacement(nSamples, this.subsample, rand);
                column = indices.map(i => X[i][j]);
            } else {
                column = X.map(row => row[j]);
            }
            const observed = column.filter(v => !Number.isNaN(v));
            if (observed.length === 0) {
                throw new Error(`Feature ${j} has no observed values`);
            }
            const sorted = observed.sort((a, b) => a - b);
            const qs = this.references.map(r => quantileSorted(sorted, r * 100));
            // enforce monotonicity against floating-point wobble
            for (let i = 1; i < qs.length; i++) {
                if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
            }
            this.quantiles.push(qs);
        }
        this.fitted = true;
    }

    private assertFittedAndShape(X: number[][], method: string): void {
        if (!this.fitted) {
            throw new Error(`QuantileTransformer must be fitted before calling ${method}`);
        }
        assertSameFeatureCount(X, this.quantiles.length, 'X has different number of features than fitted data');
    }

    public transform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'transform');
        const clipMin = normPpf(BOUNDS_THRESHOLD);
        const clipMax = normPpf(1 - BOUNDS_THRESHOLD);
        // reversed-negated copies implement sklearn's tie-symmetric double interpolation
        const negQuantiles = this.quantiles.map(qs => qs.map(q => -q).reverse());
        const negReferences = this.references.map(r => -r).reverse();
        return X.map(row => row.map((value, j) => {
            if (Number.isNaN(value)) return NaN;
            const up = interp1d(value, this.quantiles[j], this.references);
            const down = -interp1d(-value, negQuantiles[j], negReferences);
            const p = 0.5 * (up + down);
            if (this.outputDistribution === 'uniform') {
                return p;
            }
            return Math.min(clipMax, Math.max(clipMin, normPpf(p)));
        }));
    }

    public inverseTransform(X: number[][]): number[][] {
        this.assertFittedAndShape(X, 'inverseTransform');
        return X.map(row => row.map((value, j) => {
            if (Number.isNaN(value)) return NaN;
            const p = this.outputDistribution === 'normal'
                ? normCdf(value)
                : Math.min(1, Math.max(0, value));
            return interp1d(p, this.references, this.quantiles[j]);
        }));
    }
}
registerEstimator('QuantileTransformer', QuantileTransformer);

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

export interface PolynomialFeaturesProps {
    degree?: number;
    /** Only interaction terms (no powers of a single feature). */
    interactionOnly?: boolean;
    includeBias?: boolean;
}

/** Lexicographic k-combinations of {0..n-1}, itertools ordering. */
function combinationsOf(n: number, k: number, withReplacement: boolean): number[][] {
    const out: number[][] = [];
    const combo: number[] = [];
    const rec = (start: number): void => {
        if (combo.length === k) {
            out.push(combo.slice());
            return;
        }
        for (let i = start; i < n; i++) {
            combo.push(i);
            rec(withReplacement ? i : i + 1);
            combo.pop();
        }
    };
    rec(0);
    return out;
}

/**
 * Generate polynomial and interaction features with sklearn's column
 * ordering: degree-ascending, and within each degree the lexicographic
 * `itertools.combinations(_with_replacement)` order.
 */
export class PolynomialFeatures extends TransformerBase {
    private degree: number;
    private interactionOnly: boolean;
    private includeBias: boolean;
    private nInputFeatures: number;
    private combos: number[][];
    private fitted: boolean;

    constructor(props: PolynomialFeaturesProps = {}) {
        super();
        const { degree = 2, interactionOnly = false, includeBias = true } = props;
        if (!Number.isInteger(degree) || degree < 0) {
            throw new Error('degree must be a non-negative integer');
        }
        if (degree === 0 && !includeBias) {
            throw new Error('degree=0 with includeBias=false would produce an empty output');
        }
        this.degree = degree;
        this.interactionOnly = interactionOnly;
        this.includeBias = includeBias;
        this.nInputFeatures = 0;
        this.combos = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { degree: this.degree, interactionOnly: this.interactionOnly, includeBias: this.includeBias };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        this.nInputFeatures = X[0].length;
        this.combos = [];
        if (this.includeBias) {
            this.combos.push([]);
        }
        for (let d = 1; d <= this.degree; d++) {
            for (const combo of combinationsOf(this.nInputFeatures, d, !this.interactionOnly)) {
                this.combos.push(combo);
            }
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('PolynomialFeatures must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.nInputFeatures, 'X has different number of features than fitted data');
        return X.map(row => this.combos.map(combo => {
            let product = 1;
            for (const idx of combo) {
                product *= row[idx];
            }
            return product;
        }));
    }

    /** Output feature names, e.g. ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']. */
    public getFeatureNamesOut(inputNames?: string[]): string[] {
        if (!this.fitted) {
            throw new Error('PolynomialFeatures must be fitted before calling getFeatureNamesOut');
        }
        const names = inputNames ?? Array.from({ length: this.nInputFeatures }, (_, i) => `x${i}`);
        if (names.length !== this.nInputFeatures) {
            throw new Error('inputNames has different number of features than fitted data');
        }
        return this.combos.map(combo => {
            if (combo.length === 0) {
                return '1';
            }
            const parts: string[] = [];
            let i = 0;
            while (i < combo.length) {
                let power = 1;
                while (i + power < combo.length && combo[i + power] === combo[i]) {
                    power++;
                }
                parts.push(power === 1 ? names[combo[i]] : `${names[combo[i]]}^${power}`);
                i += power;
            }
            return parts.join(' ');
        });
    }
}
registerEstimator('PolynomialFeatures', PolynomialFeatures);

// ---------------------------------------------------------------------------
// KBinsDiscretizer
// ---------------------------------------------------------------------------

export interface KBinsDiscretizerProps {
    nBins?: number;
    encode?: 'ordinal' | 'onehot-dense';
    strategy?: 'uniform' | 'quantile' | 'kmeans';
}

/** Lloyd's algorithm on 1-D data from given initial centers; returns sorted centers. */
function kmeans1d(values: number[], initCenters: number[]): number[] {
    let centers = initCenters.slice();
    for (let iter = 0; iter < 300; iter++) {
        const sums = new Array(centers.length).fill(0);
        const counts = new Array(centers.length).fill(0);
        for (const v of values) {
            let best = 0;
            let bestD = Infinity;
            for (let i = 0; i < centers.length; i++) {
                const d = Math.abs(v - centers[i]);
                if (d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            sums[best] += v;
            counts[best]++;
        }
        let moved = 0;
        const next = centers.map((c, i) => (counts[i] > 0 ? sums[i] / counts[i] : c));
        for (let i = 0; i < centers.length; i++) {
            moved = Math.max(moved, Math.abs(next[i] - centers[i]));
        }
        centers = next;
        if (moved < 1e-12) break;
    }
    return centers.sort((a, b) => a - b);
}

/**
 * Bin continuous data into intervals. Strategies: 'uniform' (equal-width),
 * 'quantile' (equal-frequency, linear-interpolation percentiles), 'kmeans'
 * (1-D Lloyd initialized at uniform bin midpoints, like sklearn). Bin indices
 * are clipped to [0, nBins-1]; degenerate edges (< 1e-8 apart) are merged for
 * the quantile/kmeans strategies as sklearn does. Constant features collapse
 * to a single bin whose edges are the constant value (sklearn uses
 * [-inf, inf]; using the value keeps `inverseTransform` well-defined).
 */
export class KBinsDiscretizer extends TransformerBase {
    private nBins: number;
    private encode: 'ordinal' | 'onehot-dense';
    private strategy: 'uniform' | 'quantile' | 'kmeans';
    private binEdges: number[][];
    private nBinsPerFeature: number[];
    private fitted: boolean;

    constructor(props: KBinsDiscretizerProps = {}) {
        super();
        const { nBins = 5, encode = 'ordinal', strategy = 'uniform' } = props;
        if (!Number.isInteger(nBins) || nBins < 2) {
            throw new Error('nBins must be an integer >= 2');
        }
        if (encode !== 'ordinal' && encode !== 'onehot-dense') {
            throw new Error(`Unknown encode "${encode}". Expected 'ordinal' or 'onehot-dense'.`);
        }
        if (strategy !== 'uniform' && strategy !== 'quantile' && strategy !== 'kmeans') {
            throw new Error(`Unknown strategy "${strategy}". Expected 'uniform', 'quantile' or 'kmeans'.`);
        }
        this.nBins = nBins;
        this.encode = encode;
        this.strategy = strategy;
        this.binEdges = [];
        this.nBinsPerFeature = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { nBins: this.nBins, encode: this.encode, strategy: this.strategy };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.binEdges = [];
        this.nBinsPerFeature = [];
        for (let j = 0; j < nFeatures; j++) {
            const column = X.map(row => row[j]);
            const colMin = Math.min(...column);
            const colMax = Math.max(...column);
            if (colMin === colMax) {
                this.binEdges.push([colMin, colMax]);
                this.nBinsPerFeature.push(1);
                continue;
            }
            let edges: number[];
            if (this.strategy === 'uniform') {
                edges = Array.from({ length: this.nBins + 1 }, (_, i) => colMin + ((colMax - colMin) * i) / this.nBins);
            } else if (this.strategy === 'quantile') {
                const sorted = column.slice().sort((a, b) => a - b);
                edges = Array.from({ length: this.nBins + 1 }, (_, i) => quantileSorted(sorted, (100 * i) / this.nBins));
            } else {
                const uniform = Array.from({ length: this.nBins + 1 }, (_, i) => colMin + ((colMax - colMin) * i) / this.nBins);
                const init = Array.from({ length: this.nBins }, (_, i) => (uniform[i] + uniform[i + 1]) / 2);
                const centers = kmeans1d(column, init);
                edges = [colMin];
                for (let i = 0; i + 1 < centers.length; i++) {
                    edges.push((centers[i] + centers[i + 1]) / 2);
                }
                edges.push(colMax);
            }
            if (this.strategy !== 'uniform') {
                // merge degenerate edges (sklearn: np.ediff1d(...) > 1e-8 mask)
                const deduped = [edges[0]];
                for (let i = 1; i < edges.length; i++) {
                    if (edges[i] - deduped[deduped.length - 1] > 1e-8) {
                        deduped.push(edges[i]);
                    }
                }
                edges = deduped.length >= 2 ? deduped : [colMin, colMax];
            }
            this.binEdges.push(edges);
            this.nBinsPerFeature.push(edges.length - 1);
        }
        this.fitted = true;
    }

    private ordinalIndex(value: number, j: number): number {
        // searchsorted(edges[1:-1], value, side='right'): count of interior edges <= value
        const edges = this.binEdges[j];
        let bin = 0;
        for (let e = 1; e < edges.length - 1; e++) {
            if (value >= edges[e]) bin++;
        }
        return bin; // by construction already clipped to [0, nBins_j - 1]
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('KBinsDiscretizer must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.binEdges.length, 'X has different number of features than fitted data');
        return X.map(row => {
            const out: number[] = [];
            for (let j = 0; j < row.length; j++) {
                const bin = this.ordinalIndex(row[j], j);
                if (this.encode === 'ordinal') {
                    out.push(bin);
                } else {
                    for (let b = 0; b < this.nBinsPerFeature[j]; b++) {
                        out.push(b === bin ? 1 : 0);
                    }
                }
            }
            return out;
        });
    }

    /** Map ordinal bin indices back to bin centers. Only for encode='ordinal'. */
    public inverseTransform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('KBinsDiscretizer must be fitted before calling inverseTransform');
        }
        if (this.encode !== 'ordinal') {
            throw new Error("inverseTransform is only supported for encode='ordinal'");
        }
        assertSameFeatureCount(X, this.binEdges.length, 'X has different number of features than fitted data');
        return X.map(row => row.map((value, j) => {
            const edges = this.binEdges[j];
            const bin = Math.min(this.nBinsPerFeature[j] - 1, Math.max(0, Math.round(value)));
            return (edges[bin] + edges[bin + 1]) / 2;
        }));
    }
}
registerEstimator('KBinsDiscretizer', KBinsDiscretizer);

// ---------------------------------------------------------------------------
// KNNImputer
// ---------------------------------------------------------------------------

export interface KNNImputerProps {
    nNeighbors?: number;
    weights?: 'uniform' | 'distance';
    metric?: 'nanEuclidean';
}

/**
 * NaN-aware Euclidean distance with sklearn's missing-coordinate weighting:
 * sqrt(nFeatures / nPresent * sum of squared diffs over mutually present
 * coordinates). NaN when the rows share no present coordinate.
 */
function nanEuclideanDistance(a: number[], b: number[]): number {
    const nFeatures = a.length;
    let present = 0;
    let sq = 0;
    for (let j = 0; j < nFeatures; j++) {
        if (!Number.isNaN(a[j]) && !Number.isNaN(b[j])) {
            const d = a[j] - b[j];
            sq += d * d;
            present++;
        }
    }
    if (present === 0) return NaN;
    return Math.sqrt((nFeatures / present) * sq);
}

/**
 * Impute missing values (NaN) from the k nearest neighbors measured with the
 * nan-Euclidean metric. For each missing entry the donors are the k nearest
 * fit samples that have that feature observed (fewer donors are fine, as in
 * sklearn); when no donor exists the column mean of the fit data is used.
 */
export class KNNImputer extends TransformerBase {
    private nNeighbors: number;
    private weights: 'uniform' | 'distance';
    private metric: 'nanEuclidean';
    private fitX: number[][];
    private statistics: number[];
    private fitted: boolean;

    constructor(props: KNNImputerProps = {}) {
        super();
        const { nNeighbors = 5, weights = 'uniform', metric = 'nanEuclidean' } = props;
        if (!Number.isInteger(nNeighbors) || nNeighbors < 1) {
            throw new Error('nNeighbors must be a positive integer');
        }
        if (weights !== 'uniform' && weights !== 'distance') {
            throw new Error(`Unknown weights "${weights}". Expected 'uniform' or 'distance'.`);
        }
        if (metric !== 'nanEuclidean') {
            throw new Error(`Unknown metric "${metric}". Only 'nanEuclidean' is supported.`);
        }
        this.nNeighbors = nNeighbors;
        this.weights = weights;
        this.metric = metric;
        this.fitX = [];
        this.statistics = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { nNeighbors: this.nNeighbors, weights: this.weights, metric: this.metric };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        // sklearn drops fit samples that are missing in every feature
        this.fitX = X.filter(row => row.some(v => !Number.isNaN(v))).map(row => row.slice());
        if (this.fitX.length === 0) {
            throw new Error('X has no rows with observed values');
        }
        this.statistics = new Array(nFeatures).fill(0);
        for (let j = 0; j < nFeatures; j++) {
            const observed = observedColumn(this.fitX, j);
            if (observed.length === 0) {
                throw new Error(`Feature ${j} has no observed values`);
            }
            this.statistics[j] = meanOf(observed);
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('KNNImputer must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.statistics.length, 'X has different number of features than fitted data');
        return X.map(row => {
            if (!row.some(v => Number.isNaN(v))) {
                return row.slice();
            }
            const distances = this.fitX.map(fitRow => nanEuclideanDistance(row, fitRow));
            const out = row.slice();
            for (let j = 0; j < row.length; j++) {
                if (!Number.isNaN(row[j])) continue;
                const candidates: { dist: number; value: number; index: number }[] = [];
                for (let i = 0; i < this.fitX.length; i++) {
                    if (!Number.isNaN(this.fitX[i][j]) && !Number.isNaN(distances[i])) {
                        candidates.push({ dist: distances[i], value: this.fitX[i][j], index: i });
                    }
                }
                if (candidates.length === 0) {
                    out[j] = this.statistics[j];
                    continue;
                }
                candidates.sort((a, b) => (a.dist === b.dist ? a.index - b.index : a.dist - b.dist));
                const donors = candidates.slice(0, Math.min(this.nNeighbors, candidates.length));
                if (this.weights === 'uniform') {
                    out[j] = meanOf(donors.map(d => d.value));
                } else {
                    const zeroDist = donors.filter(d => d.dist === 0);
                    if (zeroDist.length > 0) {
                        out[j] = meanOf(zeroDist.map(d => d.value));
                    } else {
                        let wSum = 0;
                        let vSum = 0;
                        for (const d of donors) {
                            const w = 1 / d.dist;
                            wSum += w;
                            vSum += w * d.value;
                        }
                        out[j] = vSum / wSum;
                    }
                }
            }
            return out;
        });
    }
}
registerEstimator('KNNImputer', KNNImputer);

// ---------------------------------------------------------------------------
// LabelBinarizer
// ---------------------------------------------------------------------------

export interface LabelBinarizerProps {
    negLabel?: number;
    posLabel?: number;
}

/**
 * Binarize 1-D labels in a one-vs-all fashion. With exactly two classes the
 * output is a single column (sklearn behavior); with more classes it is a
 * one-hot matrix. Operates on 1-D arrays, so it extends BaseEstimator
 * directly (like LabelEncoder).
 */
export class LabelBinarizer extends BaseEstimator {
    private negLabel: number;
    private posLabel: number;
    private classes: number[];
    private fitted: boolean;

    constructor(props: LabelBinarizerProps = {}) {
        super();
        const { negLabel = 0, posLabel = 1 } = props;
        if (negLabel >= posLabel) {
            throw new Error('negLabel must be strictly less than posLabel');
        }
        this.negLabel = negLabel;
        this.posLabel = posLabel;
        this.classes = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return { negLabel: this.negLabel, posLabel: this.posLabel };
    }

    public fit(y: number[]): void {
        if (y.length === 0) {
            throw new Error('y must be non-empty');
        }
        this.classes = Array.from(new Set(y)).sort((a, b) => a - b);
        this.fitted = true;
    }

    public transform(y: number[]): number[][] {
        if (!this.fitted) {
            throw new Error('LabelBinarizer must be fitted before calling transform');
        }
        const indices = new Map<number, number>();
        this.classes.forEach((value, index) => indices.set(value, index));
        return y.map(value => {
            const index = indices.get(value);
            if (index === undefined) {
                throw new Error(`Unknown label ${value}`);
            }
            if (this.classes.length === 1) {
                return [this.negLabel];
            }
            if (this.classes.length === 2) {
                return [index === 1 ? this.posLabel : this.negLabel];
            }
            return this.classes.map((_, i) => (i === index ? this.posLabel : this.negLabel));
        });
    }

    public fitTransform(y: number[]): number[][] {
        this.fit(y);
        return this.transform(y);
    }

    public inverseTransform(Y: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('LabelBinarizer must be fitted before calling inverseTransform');
        }
        if (Y.length === 0) {
            return [];
        }
        if (this.classes.length <= 2) {
            const threshold = (this.posLabel + this.negLabel) / 2;
            return Y.map(row => {
                if (row.length !== 1) {
                    throw new Error('Expected a single-column matrix for binary labels');
                }
                if (this.classes.length === 1) {
                    return this.classes[0];
                }
                return row[0] > threshold ? this.classes[1] : this.classes[0];
            });
        }
        return Y.map(row => {
            if (row.length !== this.classes.length) {
                throw new Error('X has different number of columns than fitted classes');
            }
            let best = 0;
            for (let i = 1; i < row.length; i++) {
                if (row[i] > row[best]) best = i;
            }
            return this.classes[best];
        });
    }
}
registerEstimator('LabelBinarizer', LabelBinarizer);

// ---------------------------------------------------------------------------
// FunctionTransformer
// ---------------------------------------------------------------------------

export type ElementwiseFunc = (value: number) => number;

export interface FunctionTransformerProps {
    /**
     * Elementwise function, or the string name of a built-in ('log1p',
     * 'expm1', 'identity'). Only string names are serializable; passing a raw
     * function makes `toJSON()` throw (documented codec behavior).
     */
    func?: ElementwiseFunc | string;
    inverseFunc?: ElementwiseFunc | string;
}

const ELEMENTWISE_FUNCS: Record<string, ElementwiseFunc> = {
    log1p: Math.log1p,
    expm1: Math.expm1,
    identity: (value: number) => value,
};

/**
 * Apply a user-supplied elementwise function as a transformer. Stateless:
 * `fit` is a no-op.
 */
export class FunctionTransformer extends TransformerBase {
    private func: ElementwiseFunc | string;
    private inverseFunc: ElementwiseFunc | string;

    constructor(props: FunctionTransformerProps = {}) {
        super();
        // stored as passed (string name or raw function); resolved lazily so
        // string-configured instances stay serializable.
        this.func = props.func ?? 'identity';
        this.inverseFunc = props.inverseFunc ?? 'identity';
    }

    public getParams(): Params {
        return { func: this.func, inverseFunc: this.inverseFunc };
    }

    private resolve(func: ElementwiseFunc | string): ElementwiseFunc {
        if (typeof func !== 'string') {
            return func;
        }
        const fn = ELEMENTWISE_FUNCS[func];
        if (!fn) {
            throw new Error(`Unknown function "${func}". Built-ins: ${Object.keys(ELEMENTWISE_FUNCS).join(', ')}.`);
        }
        return fn;
    }

    public fit(_X: number[][]): void {}

    public transform(X: number[][]): number[][] {
        validateMatrix(X);
        const fn = this.resolve(this.func);
        return X.map(row => row.map(fn));
    }

    public inverseTransform(X: number[][]): number[][] {
        validateMatrix(X);
        const fn = this.resolve(this.inverseFunc);
        return X.map(row => row.map(fn));
    }
}
registerEstimator('FunctionTransformer', FunctionTransformer);

// ---------------------------------------------------------------------------
// MissingIndicator
// ---------------------------------------------------------------------------

export interface MissingIndicatorProps {
    /** 'missing-only': indicator columns only for features with missing values at fit. */
    features?: 'missing-only' | 'all';
}

/**
 * Binary (0/1) indicators for missing (NaN) values. With
 * features='missing-only' the output has one column per feature that
 * contained missing values during fit (sklearn's error_on_new check for new
 * missing features at transform time is not implemented).
 */
export class MissingIndicator extends TransformerBase {
    private features: 'missing-only' | 'all';
    private featureIndices: number[];
    private nInputFeatures: number;
    private fitted: boolean;

    constructor(props: MissingIndicatorProps = {}) {
        super();
        const { features = 'missing-only' } = props;
        if (features !== 'missing-only' && features !== 'all') {
            throw new Error(`Unknown features "${features}". Expected 'missing-only' or 'all'.`);
        }
        this.features = features;
        this.featureIndices = [];
        this.nInputFeatures = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return { features: this.features };
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.nInputFeatures = nFeatures;
        this.featureIndices = [];
        for (let j = 0; j < nFeatures; j++) {
            if (this.features === 'all' || X.some(row => Number.isNaN(row[j]))) {
                this.featureIndices.push(j);
            }
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('MissingIndicator must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.nInputFeatures, 'X has different number of features than fitted data');
        return X.map(row => this.featureIndices.map(j => (Number.isNaN(row[j]) ? 1 : 0)));
    }
}
registerEstimator('MissingIndicator', MissingIndicator);
