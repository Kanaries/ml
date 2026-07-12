import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';

export type IsotonicOutOfBounds = 'clip' | 'nan' | 'raise';

export interface IsotonicRegressionProps {
    /** Lower bound on the fitted values (sklearn `y_min`). */
    yMin?: number;
    /** Upper bound on the fitted values (sklearn `y_max`). */
    yMax?: number;
    /**
     * Whether the fitted function is non-decreasing (`true`, default),
     * non-increasing (`false`) or inferred from the sign of the Spearman rank
     * correlation between x and y (`'auto'`, sklearn `check_increasing`).
     */
    increasing?: boolean | 'auto';
    /**
     * How `predict`/`transform` handle inputs outside the training range:
     * 'clip' (default) evaluates at the nearest boundary, 'nan' returns NaN,
     * 'raise' throws.
     */
    outOfBounds?: IsotonicOutOfBounds;
}

/**
 * Normalize the accepted input shapes (`number[]` or a single-column
 * `number[][]` matrix) to a plain 1-D array.
 */
function toColumn(X: number[][] | number[], caller: string): number[] {
    if (!Array.isArray(X) || X.length === 0) {
        throw new Error(`${caller}: X must be a non-empty array`);
    }
    if (!Array.isArray(X[0])) {
        return (X as number[]).slice();
    }
    return (X as number[][]).map((row) => {
        if (!Array.isArray(row) || row.length !== 1) {
            throw new Error(`${caller}: isotonic regression is 1-D; pass number[] or an [nSamples, 1] matrix`);
        }
        return row[0];
    });
}

/** Ranks with ties averaged (as scipy.stats.rankdata(method='average')). */
function rankdata(values: number[]): number[] {
    const n = values.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
    const ranks = new Array<number>(n);
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
        const avgRank = (i + j + 2) / 2; // ranks are 1-based
        for (let k = i; k <= j; k++) ranks[order[k]] = avgRank;
        i = j + 1;
    }
    return ranks;
}

/** Spearman rank correlation; 0 when either variable is constant. */
function spearmanRho(x: number[], y: number[]): number {
    const rx = rankdata(x);
    const ry = rankdata(y);
    const n = x.length;
    const mx = rx.reduce((s, v) => s + v, 0) / n;
    const my = ry.reduce((s, v) => s + v, 0) / n;
    let cov = 0;
    let vx = 0;
    let vy = 0;
    for (let i = 0; i < n; i++) {
        const dx = rx[i] - mx;
        const dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    const denom = Math.sqrt(vx * vy);
    return denom === 0 ? 0 : cov / denom;
}

/**
 * Pool-adjacent-violators algorithm producing the weighted least-squares
 * non-decreasing fit. Direct port of scikit-learn's
 * `_inplace_contiguous_isotonic_regression` (Best & Chakravarti 1990 active
 * set, O(n)).
 */
export function pavaNonDecreasing(y: number[], w: number[]): number[] {
    const n = y.length;
    const yArr = y.slice();
    const wArr = w.slice();
    // target describes a list of blocks: for any block [i, j],
    // target[i] === j and target[j] === i.
    const target = Array.from({ length: n }, (_, i) => i);
    let i = 0;
    while (i < n) {
        let k = target[i] + 1;
        if (k === n) break;
        if (yArr[i] < yArr[k]) {
            i = k;
            continue;
        }
        let sumWy = wArr[i] * yArr[i];
        let sumW = wArr[i];
        for (;;) {
            // We are within a decreasing subsequence.
            const prevY = yArr[k];
            sumWy += wArr[k] * yArr[k];
            sumW += wArr[k];
            k = target[k] + 1;
            if (k === n || prevY < yArr[k]) {
                // Non-singleton decreasing subsequence is finished; update
                // first entry.
                yArr[i] = sumWy / sumW;
                wArr[i] = sumW;
                target[i] = k - 1;
                target[k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can: this makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[i - 1];
                }
                break;
            }
        }
    }
    // Reconstruct the solution.
    i = 0;
    while (i < n) {
        const k = target[i] + 1;
        for (let j = i + 1; j < k; j++) yArr[j] = yArr[i];
        i = k;
    }
    return yArr;
}

/**
 * Isotonic (monotonic) regression, mirroring `sklearn.isotonic.IsotonicRegression`.
 *
 * Fits a free-form, non-decreasing (or non-increasing) step-linear function to
 * 1-D data by weighted pool-adjacent-violators (PAVA), then predicts by linear
 * interpolation between the fitted thresholds.
 *
 * Input shape: the estimator is inherently 1-D, so `fit`, `predict` and
 * `transform` accept either a plain `number[]` or a single-column
 * `number[][]` matrix (so it composes with matrix-based meta-estimators).
 *
 * Matches sklearn:
 *  - ties in x are lexicographically sorted by (x, y) and merged into a single
 *    point whose y is the sample-weight-weighted mean (sklearn `_make_unique`);
 *  - `increasing='auto'` picks the direction from the sign of the Spearman
 *    rank correlation (sklearn `check_increasing`, `rho >= 0` → increasing);
 *  - `yMin`/`yMax` clip the fitted values after PAVA;
 *  - `outOfBounds` = 'clip' | 'nan' | 'raise' as in sklearn.
 *
 * Omitted vs sklearn: the redundant-threshold trimming sklearn applies before
 * building its interpolator (`_build_y` keeps only points where the fit
 * changes slope). We keep every unique-x point; the interpolated predictions
 * are identical. The `increasing='auto'` confidence-interval warning is also
 * omitted.
 */
export class IsotonicRegression extends RegressorBase {
    private yMin?: number;
    private yMax?: number;
    private increasing: boolean | 'auto';
    private outOfBounds: IsotonicOutOfBounds;

    /** Unique ascending x thresholds of the fitted function. */
    private xThresholds: number[] = [];
    /** Fitted values aligned with `xThresholds`. */
    private yThresholds: number[] = [];
    /** Direction actually used at fit time (resolves 'auto'). */
    public increasingFitted: boolean | null = null;
    private fitted = false;

    constructor(props: IsotonicRegressionProps = {}) {
        super();
        const { yMin, yMax, increasing = true, outOfBounds = 'clip' } = props;
        if (increasing !== true && increasing !== false && increasing !== 'auto') {
            throw new Error("increasing must be true, false or 'auto'");
        }
        if (outOfBounds !== 'clip' && outOfBounds !== 'nan' && outOfBounds !== 'raise') {
            throw new Error("outOfBounds must be 'clip', 'nan' or 'raise'");
        }
        if (yMin !== undefined && !Number.isFinite(yMin)) throw new Error('yMin must be a finite number');
        if (yMax !== undefined && !Number.isFinite(yMax)) throw new Error('yMax must be a finite number');
        if (yMin !== undefined && yMax !== undefined && yMin > yMax) {
            throw new Error('yMin must be <= yMax');
        }
        this.yMin = yMin;
        this.yMax = yMax;
        this.increasing = increasing;
        this.outOfBounds = outOfBounds;
    }

    public getParams(): Params {
        return {
            yMin: this.yMin,
            yMax: this.yMax,
            increasing: this.increasing,
            outOfBounds: this.outOfBounds,
        };
    }

    public fit(X: number[][] | number[], y: number[], sampleWeight?: number[]): void {
        const x = toColumn(X, 'IsotonicRegression.fit');
        if (!Array.isArray(y) || y.length !== x.length) {
            throw new Error('X and y must be non-empty arrays of the same length');
        }
        let w: number[];
        if (sampleWeight !== undefined) {
            if (sampleWeight.length !== x.length) {
                throw new Error('sampleWeight must have the same length as X');
            }
            if (sampleWeight.some((v) => !Number.isFinite(v) || v <= 0)) {
                throw new Error('sampleWeight entries must be finite and > 0');
            }
            w = sampleWeight.slice();
        } else {
            w = new Array(x.length).fill(1);
        }

        const inc = this.increasing === 'auto' ? spearmanRho(x, y) >= 0 : this.increasing;

        // Lexicographic sort by (x, y) — sample-weight-free secondary sort,
        // as in sklearn (np.lexsort((y, X))).
        const order = Array.from({ length: x.length }, (_, i) => i).sort((a, b) => x[a] - x[b] || y[a] - y[b]);
        const xs = order.map((i) => x[i]);
        const ys = order.map((i) => y[i]);
        const ws = order.map((i) => w[i]);

        // sklearn _make_unique: merge duplicate x into one point with the
        // weighted-average y and summed weight.
        const ux: number[] = [];
        const uy: number[] = [];
        const uw: number[] = [];
        let i = 0;
        while (i < xs.length) {
            const xv = xs[i];
            let sumW = 0;
            let sumWy = 0;
            let j = i;
            while (j < xs.length && xs[j] === xv) {
                sumW += ws[j];
                sumWy += ws[j] * ys[j];
                j++;
            }
            ux.push(xv);
            uy.push(sumWy / sumW);
            uw.push(sumW);
            i = j;
        }

        // Non-increasing fits run PAVA on the reversed sequence.
        let fittedY: number[];
        if (inc) {
            fittedY = pavaNonDecreasing(uy, uw);
        } else {
            fittedY = pavaNonDecreasing(uy.slice().reverse(), uw.slice().reverse()).reverse();
        }
        if (this.yMin !== undefined || this.yMax !== undefined) {
            const lo = this.yMin ?? -Infinity;
            const hi = this.yMax ?? Infinity;
            fittedY = fittedY.map((v) => Math.min(Math.max(v, lo), hi));
        }

        this.xThresholds = ux;
        this.yThresholds = fittedY;
        this.increasingFitted = inc;
        this.fitted = true;
    }

    /** Piecewise-linear interpolation over the fitted thresholds. */
    private interpolate(value: number): number {
        const xs = this.xThresholds;
        const ys = this.yThresholds;
        const n = xs.length;
        let t = value;
        if (t < xs[0] || t > xs[n - 1]) {
            switch (this.outOfBounds) {
                case 'raise':
                    throw new Error(
                        `input value ${t} is outside the training range [${xs[0]}, ${xs[n - 1]}] ` +
                            "and outOfBounds='raise'",
                    );
                case 'nan':
                    return NaN;
                case 'clip':
                    t = Math.min(Math.max(t, xs[0]), xs[n - 1]);
            }
        }
        if (n === 1) return ys[0];
        // binary search for the segment [lo, lo + 1] containing t
        let lo = 0;
        let hi = n - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (xs[mid] <= t) lo = mid;
            else hi = mid;
        }
        if (t === xs[lo]) return ys[lo];
        if (t === xs[hi]) return ys[hi];
        const frac = (t - xs[lo]) / (xs[hi] - xs[lo]);
        return ys[lo] + frac * (ys[hi] - ys[lo]);
    }

    public predict(X: number[][] | number[]): number[] {
        if (!this.fitted) {
            throw new Error('IsotonicRegression must be fitted before calling predict');
        }
        return toColumn(X, 'IsotonicRegression.predict').map((v) => this.interpolate(v));
    }

    /** Alias of `predict` (sklearn exposes both). */
    public transform(X: number[][] | number[]): number[] {
        return this.predict(X);
    }
}
registerEstimator('IsotonicRegression', IsotonicRegression);
