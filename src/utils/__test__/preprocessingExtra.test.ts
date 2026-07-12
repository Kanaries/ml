/**
 * Tests for the additional preprocessing transformers. Expected values are
 * verified against scikit-learn 1.4 (the exact call is quoted above each
 * assertion block).
 */
import {
    FunctionTransformer,
    KBinsDiscretizer,
    KNNImputer,
    LabelBinarizer,
    MissingIndicator,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
} from '../preprocessing';

/** Population skewness (Fisher-Pearson, ddof=0), like scipy.stats.skew. */
function skewness(values: number[]): number {
    const n = values.length;
    const mean = values.reduce((s, v) => s + v, 0) / n;
    let m2 = 0;
    let m3 = 0;
    for (const v of values) {
        const d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    return m3 / Math.pow(m2, 1.5);
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

test('RobustScaler centers by median and scales by IQR (numpy linear quantiles)', () => {
    // sklearn: RobustScaler().fit_transform([[1.], [3.], [5.], [7.]])
    // median=4, q25=2.5, q75=5.5, IQR=3 -> [[-1.], [-1/3], [1/3], [1.]]
    const scaler = new RobustScaler();
    const out = scaler.fitTransform([[1], [3], [5], [7]]);
    expect(out[0][0]).toBeCloseTo(-1, 10);
    expect(out[1][0]).toBeCloseTo(-1 / 3, 10);
    expect(out[2][0]).toBeCloseTo(1 / 3, 10);
    expect(out[3][0]).toBeCloseTo(1, 10);
});

test('RobustScaler is robust to outliers and handles multiple columns', () => {
    // sklearn: RobustScaler().fit_transform([[1,-2],[2,-1],[3,0],[4,1],[100,2]])
    // col0: median=3, q25=2, q75=4 -> [-1, -0.5, 0, 0.5, 48.5]
    // col1: median=0, q25=-1, q75=1 -> [-1, -0.5, 0, 0.5, 1]
    const X = [[1, -2], [2, -1], [3, 0], [4, 1], [100, 2]];
    const out = new RobustScaler().fitTransform(X);
    expect(out.map(r => r[0])).toEqual([-1, -0.5, 0, 0.5, 48.5]);
    expect(out.map(r => r[1])).toEqual([-1, -0.5, 0, 0.5, 1]);
});

test('RobustScaler honors quantileRange, withCentering and withScaling', () => {
    const X = [[1], [2], [3], [4], [100]];
    // sklearn: RobustScaler(quantile_range=(10, 90)).fit_transform(X)
    // q10 = 1.4, q90 = 61.6 (numpy linear interpolation), scale = 60.2
    const ranged = new RobustScaler({ quantileRange: [10, 90] });
    expect(ranged.fitTransform(X)[4][0]).toBeCloseTo((100 - 3) / 60.2, 10);

    // sklearn: RobustScaler(with_centering=False).fit_transform(X) -> x / 2
    const noCenter = new RobustScaler({ withCentering: false });
    expect(noCenter.fitTransform(X)[0][0]).toBeCloseTo(0.5, 10);

    // sklearn: RobustScaler(with_scaling=False).fit_transform(X) -> x - 3
    const noScale = new RobustScaler({ withScaling: false });
    expect(noScale.fitTransform(X)[4][0]).toBeCloseTo(97, 10);
});

test('RobustScaler unitVariance divides the scale by the normal quantile span', () => {
    // sklearn: RobustScaler(unit_variance=True).fit_transform([[-2],[-1],[0],[1],[2]])
    // IQR = 2, adjust = norm.ppf(0.75) - norm.ppf(0.25) = 1.3489795003921634
    // -> scale = 1.4826022185056018, transformed max = 2 / scale = 1.3489795
    const scaler = new RobustScaler({ unitVariance: true });
    const out = scaler.fitTransform([[-2], [-1], [0], [1], [2]]);
    expect(out[4][0]).toBeCloseTo(1.3489795003921634, 6);
});

test('RobustScaler ignores NaN in fit statistics and passes NaN through', () => {
    // sklearn: RobustScaler().fit_transform([[1],[np.nan],[3],[5]])
    // stats from [1,3,5]: median=3, q25=2, q75=4, IQR=2
    const out = new RobustScaler().fitTransform([[1], [NaN], [3], [5]]);
    expect(out[0][0]).toBeCloseTo(-1, 10);
    expect(Number.isNaN(out[1][0])).toBe(true);
    expect(out[2][0]).toBeCloseTo(0, 10);
    expect(out[3][0]).toBeCloseTo(1, 10);
});

test('RobustScaler inverseTransform recovers the original data', () => {
    const X = [[1, -2], [2, -1], [3, 0], [4, 1], [100, 2]];
    const scaler = new RobustScaler({ unitVariance: true });
    const restored = scaler.inverseTransform(scaler.fitTransform(X));
    for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
            expect(restored[i][j]).toBeCloseTo(X[i][j], 8);
        }
    }
});

test('RobustScaler validates quantileRange and fitted state', () => {
    expect(() => new RobustScaler({ quantileRange: [75, 25] })).toThrow(/quantileRange/);
    expect(() => new RobustScaler().transform([[1]])).toThrow(/must be fitted/);
});

// ---------------------------------------------------------------------------
// PowerTransformer
// ---------------------------------------------------------------------------

test('PowerTransformer yeo-johnson matches the sklearn docstring example', () => {
    // sklearn: PowerTransformer().fit_transform([[1, 2], [3, 2], [4, 5]])
    // -> [[-1.31616039, -0.70710678], [0.20998268, -0.70710678], [1.10617771, 1.41421356]]
    const pt = new PowerTransformer();
    const out = pt.fitTransform([[1, 2], [3, 2], [4, 5]]);
    expect(out[0][0]).toBeCloseTo(-1.31616039, 4);
    expect(out[1][0]).toBeCloseTo(0.20998268, 4);
    expect(out[2][0]).toBeCloseTo(1.10617771, 4);
    expect(out[0][1]).toBeCloseTo(-0.70710678, 6);
    expect(out[1][1]).toBeCloseTo(-0.70710678, 6);
    expect(out[2][1]).toBeCloseTo(1.41421356, 6);
});

test('PowerTransformer box-cox removes the skew of log-normal data', () => {
    // x = exp(z) for symmetric z: box-cox MLE lambda ~ 0 (log transform),
    // so the transformed sample is approximately z and has ~zero skewness.
    const z = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    const X = z.map(v => [Math.exp(v)]);
    expect(skewness(X.map(r => r[0]))).toBeGreaterThan(1);
    const pt = new PowerTransformer({ method: 'box-cox', standardize: false });
    const out = pt.fitTransform(X).map(r => r[0]);
    expect(Math.abs(skewness(out))).toBeLessThan(0.2);
});

test('PowerTransformer yeo-johnson reduces skew of right-skewed data', () => {
    const raw = [0.1, 0.2, 0.4, 0.7, 1.1, 1.9, 3.2, 5.5, 9.1, 15.2];
    expect(skewness(raw)).toBeGreaterThan(1);
    const pt = new PowerTransformer({ method: 'yeo-johnson' });
    const out = pt.fitTransform(raw.map(v => [v])).map(r => r[0]);
    expect(Math.abs(skewness(out))).toBeLessThan(0.3);
    // standardize=true -> zero mean, unit (population) std
    const mean = out.reduce((s, v) => s + v, 0) / out.length;
    const std = Math.sqrt(out.reduce((s, v) => s + (v - mean) ** 2, 0) / out.length);
    expect(mean).toBeCloseTo(0, 8);
    expect(std).toBeCloseTo(1, 8);
});

test('PowerTransformer box-cox requires strictly positive data', () => {
    expect(() => new PowerTransformer({ method: 'box-cox' }).fit([[1], [0]]))
        .toThrow(/strictly positive/);
    expect(() => new PowerTransformer({ method: 'box-cox' }).fit([[2], [-1]]))
        .toThrow(/strictly positive/);
});

test('PowerTransformer inverseTransform recovers the original data (both methods)', () => {
    const positive = [[0.5], [1], [2], [4], [9]];
    const mixed = [[-3], [-0.5], [0], [1.5], [6]];
    for (const [method, X] of [['box-cox', positive], ['yeo-johnson', mixed]] as const) {
        const pt = new PowerTransformer({ method, standardize: true });
        const restored = pt.inverseTransform(pt.fitTransform(X));
        for (let i = 0; i < X.length; i++) {
            expect(restored[i][0]).toBeCloseTo(X[i][0], 6);
        }
    }
});

test('PowerTransformer ignores NaN when fitting and passes NaN through', () => {
    const pt = new PowerTransformer();
    const out = pt.fitTransform([[1], [NaN], [3], [4]]);
    expect(Number.isNaN(out[1][0])).toBe(true);
    expect(Number.isFinite(out[0][0])).toBe(true);
});

// ---------------------------------------------------------------------------
// QuantileTransformer
// ---------------------------------------------------------------------------

test('QuantileTransformer maps a uniform ramp to [0, 1]', () => {
    // sklearn: QuantileTransformer(n_quantiles=100).fit(np.arange(100).reshape(-1, 1))
    // nQuantiles=1000 silently clamps to nSamples=100; quantiles are exactly 0..99.
    const X = Array.from({ length: 100 }, (_, i) => [i]);
    const qt = new QuantileTransformer();
    qt.fit(X);
    expect(qt.transform([[0]])[0][0]).toBeCloseTo(0, 10);
    expect(qt.transform([[99]])[0][0]).toBeCloseTo(1, 10);
    // sklearn: qt.transform([[25]]) -> 0.25252525 (= 25/99)
    expect(qt.transform([[25]])[0][0]).toBeCloseTo(25 / 99, 10);
    expect(qt.transform([[49.5]])[0][0]).toBeCloseTo(0.5, 10);
    // out-of-range values clamp to the ends
    expect(qt.transform([[-10]])[0][0]).toBeCloseTo(0, 10);
    expect(qt.transform([[1000]])[0][0]).toBeCloseTo(1, 10);
    // inverse
    expect(qt.inverseTransform([[0.5]])[0][0]).toBeCloseTo(49.5, 8);
    expect(qt.inverseTransform([[25 / 99]])[0][0]).toBeCloseTo(25, 8);
});

test('QuantileTransformer averages the tie-symmetric interpolations (sklearn parity)', () => {
    // sklearn: QuantileTransformer(n_quantiles=4).fit_transform([[1],[1],[1],[3]])
    // -> [[1/3], [1/3], [1/3], [1.]]; transform([[2]]) -> 5/6
    const qt = new QuantileTransformer({ nQuantiles: 4 });
    const out = qt.fitTransform([[1], [1], [1], [3]]);
    expect(out[0][0]).toBeCloseTo(1 / 3, 10);
    expect(out[1][0]).toBeCloseTo(1 / 3, 10);
    expect(out[2][0]).toBeCloseTo(1 / 3, 10);
    expect(out[3][0]).toBeCloseTo(1, 10);
    expect(qt.transform([[2]])[0][0]).toBeCloseTo(5 / 6, 10);
});

test('QuantileTransformer normal output uses the inverse normal CDF with sklearn clipping', () => {
    const X = Array.from({ length: 100 }, (_, i) => [i]);
    const qt = new QuantileTransformer({ outputDistribution: 'normal' });
    qt.fit(X);
    // median maps to norm.ppf(0.5) = 0
    expect(qt.transform([[49.5]])[0][0]).toBeCloseTo(0, 8);
    // sklearn clips at norm.ppf(BOUNDS_THRESHOLD=1e-7) = -5.199337582
    expect(qt.transform([[0]])[0][0]).toBeCloseTo(-5.19934, 3);
    expect(qt.transform([[99]])[0][0]).toBeCloseTo(5.19934, 3);
    // scipy: norm.ppf(75/99) = 0.698527
    expect(qt.transform([[75]])[0][0]).toBeCloseTo(0.698527, 4);
    // monotone
    const a = qt.transform([[20]])[0][0];
    const b = qt.transform([[60]])[0][0];
    expect(a).toBeLessThan(b);
});

test('QuantileTransformer normal round-trips interior values', () => {
    const X = Array.from({ length: 100 }, (_, i) => [i]);
    const qt = new QuantileTransformer({ outputDistribution: 'normal' });
    qt.fit(X);
    for (const v of [5, 20, 49.5, 77, 95]) {
        const restored = qt.inverseTransform(qt.transform([[v]]))[0][0];
        expect(restored).toBeCloseTo(v, 4);
    }
});

test('QuantileTransformer subsampling is deterministic under randomState', () => {
    const rand = (seed: number) => {
        let s = seed;
        return () => {
            s = (s * 16807) % 2147483647;
            return s / 2147483647;
        };
    };
    const gen = rand(123);
    const X = Array.from({ length: 200 }, () => [gen() * 10]);
    const a = new QuantileTransformer({ nQuantiles: 20, subsample: 50, randomState: 7 });
    const b = new QuantileTransformer({ nQuantiles: 20, subsample: 50, randomState: 7 });
    expect(a.fitTransform(X)).toEqual(b.fitTransform(X));
});

test('QuantileTransformer ignores NaN in fit and passes NaN through', () => {
    const qt = new QuantileTransformer({ nQuantiles: 3 });
    const out = qt.fitTransform([[1], [NaN], [2], [3]]);
    expect(out[0][0]).toBeCloseTo(0, 10);
    expect(Number.isNaN(out[1][0])).toBe(true);
    expect(out[3][0]).toBeCloseTo(1, 10);
    expect(Number.isNaN(qt.inverseTransform([[NaN]])[0][0])).toBe(true);
});

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

test('PolynomialFeatures degree=2 uses sklearn column ordering', () => {
    // sklearn: PolynomialFeatures(2).fit_transform([[2, 3]])
    // -> [[1, 2, 3, 4, 6, 9]]
    const poly = new PolynomialFeatures({ degree: 2 });
    expect(poly.fitTransform([[2, 3]])).toEqual([[1, 2, 3, 4, 6, 9]]);
    // sklearn: poly.get_feature_names_out() -> ['1','x0','x1','x0^2','x0 x1','x1^2']
    expect(poly.getFeatureNamesOut()).toEqual(['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']);
    expect(poly.getFeatureNamesOut(['a', 'b'])).toEqual(['1', 'a', 'b', 'a^2', 'a b', 'b^2']);
});

test('PolynomialFeatures interactionOnly drops pure powers', () => {
    // sklearn: PolynomialFeatures(2, interaction_only=True).fit_transform([[2, 3, 5]])
    // -> [[1, 2, 3, 5, 6, 10, 15]]
    const poly = new PolynomialFeatures({ degree: 2, interactionOnly: true });
    expect(poly.fitTransform([[2, 3, 5]])).toEqual([[1, 2, 3, 5, 6, 10, 15]]);
    expect(poly.getFeatureNamesOut()).toEqual(['1', 'x0', 'x1', 'x2', 'x0 x1', 'x0 x2', 'x1 x2']);
});

test('PolynomialFeatures respects includeBias and higher degree', () => {
    // sklearn: PolynomialFeatures(3, include_bias=False).fit_transform([[2]]) -> [[2, 4, 8]]
    const noBias = new PolynomialFeatures({ degree: 3, includeBias: false });
    expect(noBias.fitTransform([[2]])).toEqual([[2, 4, 8]]);
    expect(noBias.getFeatureNamesOut()).toEqual(['x0', 'x0^2', 'x0^3']);

    // sklearn: PolynomialFeatures(3).fit_transform([[2, 3]])[0]
    // -> [1, 2, 3, 4, 6, 9, 8, 12, 18, 27]
    const cubic = new PolynomialFeatures({ degree: 3 });
    expect(cubic.fitTransform([[2, 3]])).toEqual([[1, 2, 3, 4, 6, 9, 8, 12, 18, 27]]);
    expect(cubic.getFeatureNamesOut()).toEqual([
        '1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2', 'x0^3', 'x0^2 x1', 'x0 x1^2', 'x1^3',
    ]);
});

test('PolynomialFeatures validates params and fitted state', () => {
    expect(() => new PolynomialFeatures({ degree: -1 })).toThrow(/non-negative/);
    expect(() => new PolynomialFeatures({ degree: 0, includeBias: false })).toThrow(/empty output/);
    expect(() => new PolynomialFeatures().transform([[1]])).toThrow(/must be fitted/);
    const poly = new PolynomialFeatures();
    poly.fit([[1, 2]]);
    expect(() => poly.transform([[1]])).toThrow(/different number of features/);
    expect(() => poly.getFeatureNamesOut(['onlyOne'])).toThrow(/different number of features/);
});

// ---------------------------------------------------------------------------
// KBinsDiscretizer
// ---------------------------------------------------------------------------

test('KBinsDiscretizer uniform strategy matches the sklearn docstring example', () => {
    // sklearn: KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    //   X = [[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]]
    //   Xt = [[0,0,0,0],[1,1,1,0],[2,2,2,1],[2,2,2,2]]
    //   inverse_transform(Xt)[0] = [-1.5, 1.5, -3.5, -0.5]
    const X = [[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]];
    const est = new KBinsDiscretizer({ nBins: 3, encode: 'ordinal', strategy: 'uniform' });
    const Xt = est.fitTransform(X);
    expect(Xt).toEqual([[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]]);
    expect(est.inverseTransform(Xt)[0]).toEqual([-1.5, 1.5, -3.5, -0.5]);
});

test('KBinsDiscretizer quantile strategy uses linear-interpolation percentile edges', () => {
    // sklearn: KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    //   on np.arange(10): bin_edges_ = [0, 1.8, 3.6, 5.4, 7.2, 9]
    //   -> bins [0,0,1,1,2,2,3,3,4,4]
    const X = Array.from({ length: 10 }, (_, i) => [i]);
    const est = new KBinsDiscretizer({ nBins: 5, encode: 'ordinal', strategy: 'quantile' });
    const Xt = est.fitTransform(X);
    expect(Xt.map(r => r[0])).toEqual([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
});

test('KBinsDiscretizer kmeans strategy uses 1-D Lloyd with uniform midpoint init', () => {
    // sklearn: KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
    //   on [[0],[1],[9],[10]]: centers -> [0.5, 9.5], bin_edges_ = [0, 5, 10]
    //   transform -> [[0],[0],[1],[1]]; inverse -> [[2.5],[2.5],[7.5],[7.5]]
    const X = [[0], [1], [9], [10]];
    const est = new KBinsDiscretizer({ nBins: 2, encode: 'ordinal', strategy: 'kmeans' });
    const Xt = est.fitTransform(X);
    expect(Xt).toEqual([[0], [0], [1], [1]]);
    expect(est.inverseTransform(Xt)).toEqual([[2.5], [2.5], [7.5], [7.5]]);
});

test('KBinsDiscretizer clips out-of-range values to the outer bins', () => {
    const est = new KBinsDiscretizer({ nBins: 2, encode: 'ordinal', strategy: 'uniform' });
    est.fit([[0], [10]]);
    expect(est.transform([[-100], [100]])).toEqual([[0], [1]]);
});

test('KBinsDiscretizer onehot-dense expands one column per bin', () => {
    // sklearn: KBinsDiscretizer(n_bins=2, encode='onehot-dense', strategy='uniform')
    //   .fit_transform([[0],[5],[10]]) -> [[1,0],[0,1],[0,1]] (edge value 5 goes right)
    const est = new KBinsDiscretizer({ nBins: 2, encode: 'onehot-dense', strategy: 'uniform' });
    expect(est.fitTransform([[0], [5], [10]])).toEqual([[1, 0], [0, 1], [0, 1]]);
    expect(() => est.inverseTransform([[1, 0]])).toThrow(/ordinal/);
});

test('KBinsDiscretizer collapses constant features to a single bin', () => {
    const est = new KBinsDiscretizer({ nBins: 3, encode: 'ordinal', strategy: 'uniform' });
    const Xt = est.fitTransform([[7, 0], [7, 5], [7, 10]]);
    expect(Xt.map(r => r[0])).toEqual([0, 0, 0]);
    expect(est.inverseTransform(Xt).map(r => r[0])).toEqual([7, 7, 7]);
});

test('KBinsDiscretizer validates params', () => {
    expect(() => new KBinsDiscretizer({ nBins: 1 })).toThrow(/nBins/);
    expect(() => new KBinsDiscretizer().transform([[1]])).toThrow(/must be fitted/);
});

// ---------------------------------------------------------------------------
// KNNImputer
// ---------------------------------------------------------------------------

test('KNNImputer matches the sklearn docstring example', () => {
    // sklearn: KNNImputer(n_neighbors=2).fit_transform(
    //   [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]])
    // -> [[1, 2, 4], [3, 4, 3], [5.5, 6, 5], [8, 8, 7]]
    // Hand check row 0, col 2 (nan-euclidean, weight nFeatures/nPresent):
    //   d(r0,r1) = sqrt(3/2 * (2^2 + 2^2)) = 3.4641
    //   d(r0,r2) = sqrt(3/1 * 4^2)         = 6.9282
    //   d(r0,r3) = sqrt(3/2 * (7^2 + 6^2)) = 11.2916
    //   donors for col2 = rows 1, 2 -> mean(3, 5) = 4
    const X = [[1, 2, NaN], [3, 4, 3], [NaN, 6, 5], [8, 8, 7]];
    const imputer = new KNNImputer({ nNeighbors: 2 });
    const out = imputer.fitTransform(X);
    expect(out[0]).toEqual([1, 2, 4]);
    expect(out[1]).toEqual([3, 4, 3]);
    expect(out[2]).toEqual([5.5, 6, 5]);
    expect(out[3]).toEqual([8, 8, 7]);
});

test('KNNImputer distance weighting weights donors by inverse distance', () => {
    // sklearn: KNNImputer(n_neighbors=2, weights='distance').fit_transform(
    //   [[1, nan], [2, 0], [4, 6]])
    // distances from row0: sqrt(2*1)=1.4142 to row1, sqrt(2*9)=4.2426 to row2
    // weighted value = (0/1.4142 + 6/4.2426) / (1/1.4142 + 1/4.2426) = 1.5
    const out = new KNNImputer({ nNeighbors: 2, weights: 'distance' })
        .fitTransform([[1, NaN], [2, 0], [4, 6]]);
    expect(out[0][1]).toBeCloseTo(1.5, 10);
});

test('KNNImputer falls back to the column mean when no donor exists', () => {
    // A receiver with no observed coordinate has NaN distance to every fit
    // row, so sklearn falls back to the fit column means: [2, 3].
    const imputer = new KNNImputer({ nNeighbors: 3 });
    imputer.fit([[1, 2], [3, 4]]);
    expect(imputer.transform([[NaN, NaN]])).toEqual([[2, 3]]);
});

test('KNNImputer leaves complete rows untouched and validates input', () => {
    const imputer = new KNNImputer({ nNeighbors: 1 });
    imputer.fit([[1, 10], [2, 20], [3, 30]]);
    expect(imputer.transform([[2.1, 19]])).toEqual([[2.1, 19]]);
    // nearest neighbor of [2.1, NaN] is [2, 20]
    expect(imputer.transform([[2.1, NaN]])).toEqual([[2.1, 20]]);
    expect(() => new KNNImputer({ nNeighbors: 0 })).toThrow(/nNeighbors/);
    expect(() => new KNNImputer().transform([[1]])).toThrow(/must be fitted/);
    expect(() => new KNNImputer().fit([[NaN], [NaN]])).toThrow(/observed values/);
});

// ---------------------------------------------------------------------------
// LabelBinarizer
// ---------------------------------------------------------------------------

test('LabelBinarizer one-hot encodes multiclass labels (sklearn docstring)', () => {
    // sklearn: lb = LabelBinarizer(); lb.fit([1, 2, 6, 4, 2])
    //   lb.classes_ -> [1, 2, 4, 6]
    //   lb.transform([1, 6]) -> [[1, 0, 0, 0], [0, 0, 0, 1]]
    const lb = new LabelBinarizer();
    lb.fit([1, 2, 6, 4, 2]);
    expect(lb.transform([1, 6])).toEqual([[1, 0, 0, 0], [0, 0, 0, 1]]);
    expect(lb.inverseTransform([[1, 0, 0, 0], [0, 0, 0, 1]])).toEqual([1, 6]);
});

test('LabelBinarizer emits a single column for binary labels (sklearn behavior)', () => {
    // sklearn: LabelBinarizer().fit_transform([0, 1, 1]) -> [[0], [1], [1]]
    const lb = new LabelBinarizer();
    expect(lb.fitTransform([0, 1, 1])).toEqual([[0], [1], [1]]);
    expect(lb.inverseTransform([[0], [1], [1]])).toEqual([0, 1, 1]);
});

test('LabelBinarizer honors negLabel/posLabel', () => {
    // sklearn: LabelBinarizer(neg_label=-1, pos_label=1).fit_transform([0, 1, 1])
    // -> [[-1], [1], [1]]
    const lb = new LabelBinarizer({ negLabel: -1, posLabel: 1 });
    expect(lb.fitTransform([0, 1, 1])).toEqual([[-1], [1], [1]]);
    expect(lb.inverseTransform([[-1], [1]])).toEqual([0, 1]);

    const multi = new LabelBinarizer({ negLabel: -1, posLabel: 2 });
    expect(multi.fitTransform([5, 6, 7])).toEqual([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]);
    expect(() => new LabelBinarizer({ negLabel: 1, posLabel: 0 })).toThrow(/negLabel/);
});

test('LabelBinarizer rejects unknown labels and unfitted use', () => {
    const lb = new LabelBinarizer();
    expect(() => lb.transform([0])).toThrow(/must be fitted/);
    lb.fit([0, 1, 2]);
    expect(() => lb.transform([9])).toThrow(/Unknown label/);
});

// ---------------------------------------------------------------------------
// FunctionTransformer
// ---------------------------------------------------------------------------

test('FunctionTransformer applies built-in functions by name', () => {
    // sklearn: FunctionTransformer(np.log1p).transform([[0, e - 1]]) -> [[0, 1]]
    const ft = new FunctionTransformer({ func: 'log1p', inverseFunc: 'expm1' });
    const out = ft.fitTransform([[0, Math.E - 1]]);
    expect(out[0][0]).toBeCloseTo(0, 12);
    expect(out[0][1]).toBeCloseTo(1, 12);
    const restored = ft.inverseTransform(out);
    expect(restored[0][0]).toBeCloseTo(0, 12);
    expect(restored[0][1]).toBeCloseTo(Math.E - 1, 12);
});

test('FunctionTransformer defaults to identity and accepts raw functions', () => {
    expect(new FunctionTransformer().fitTransform([[1, 2]])).toEqual([[1, 2]]);
    const ft = new FunctionTransformer({ func: (v: number) => v * 2, inverseFunc: (v: number) => v / 2 });
    expect(ft.fitTransform([[1, 2]])).toEqual([[2, 4]]);
    expect(ft.inverseTransform([[2, 4]])).toEqual([[1, 2]]);
});

test('FunctionTransformer rejects unknown built-in names at use time', () => {
    const ft = new FunctionTransformer({ func: 'notAFunc' });
    expect(() => ft.transform([[1]])).toThrow(/Unknown function/);
});

// ---------------------------------------------------------------------------
// MissingIndicator
// ---------------------------------------------------------------------------

test('MissingIndicator matches the sklearn docstring example', () => {
    // sklearn: MissingIndicator().fit([[nan, 1, 3], [4, 0, nan], [8, 1, 0]])
    //   features_ -> [0, 2]
    //   transform([[5, 1, nan], [nan, 2, 3], [2, 4, 0]])
    //   -> [[False, True], [True, False], [False, False]]
    const indicator = new MissingIndicator();
    indicator.fit([[NaN, 1, 3], [4, 0, NaN], [8, 1, 0]]);
    expect(indicator.transform([[5, 1, NaN], [NaN, 2, 3], [2, 4, 0]]))
        .toEqual([[0, 1], [1, 0], [0, 0]]);
});

test('MissingIndicator features="all" keeps every column', () => {
    const indicator = new MissingIndicator({ features: 'all' });
    indicator.fit([[1, NaN], [2, 3]]);
    expect(indicator.transform([[NaN, NaN], [1, 2]])).toEqual([[1, 1], [0, 0]]);
});

test('MissingIndicator validates fitted state and shape', () => {
    expect(() => new MissingIndicator().transform([[1]])).toThrow(/must be fitted/);
    const indicator = new MissingIndicator();
    indicator.fit([[1, 2]]);
    expect(() => indicator.transform([[1]])).toThrow(/different number of features/);
});
