'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Linear } from '@kanaries/ml';
import styles from './interactive.module.css';
import { DataInput, parseNumericTable } from './DataInput';
import { StatCards, formatMetric } from './StatCards';
import { CodeTabs } from './CodeTabs';
import { downloadSvgAsPng, downloadText, readUrlToolState, writeUrlToolState } from './clientUtils';
import {
  computeRegressionMetrics,
  standardizeMatrix,
  unscaleLinearWeights,
} from './calculatorMetrics';

const SIMPLE_AD_SPEND = `ad_spend_k,revenue_k
2,14.2
4,21.5
6,27.8
8,33.1
10,41.4
12,46.0
14,54.8
16,59.2
18,68.5
20,73.0
22,80.4
24,86.1`;

const MULTIVARIATE_HOUSING = `sqft,bedrooms,age_years,price_k
1100,2,25,215
1350,3,18,268
1500,3,12,305
1650,3,30,295
1800,4,8,365
2000,4,15,388
2200,4,5,442
2450,4,20,455
2600,5,10,508
2900,5,4,565
3100,5,14,580
3400,5,6,640`;

const COLLINEAR_DEMO = `sessions_k,pageviews_k,bounce_pct,signups
10,20.2,48,54
14,27.9,44,72
18,36.1,41,91
22,44.0,39,108
26,52.2,35,127
30,59.8,33,144
34,68.1,30,165
38,76.0,28,182
42,83.9,26,201
46,92.1,24,220`;

type RegressionUrlState = {
  csv?: string;
  mode?: 'ols' | 'ridge';
  alpha?: number;
  standardize?: boolean;
};

function niceRange(values: number[]): [number, number] {
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

export function LinearRegressionCalculator() {
  const [csv, setCsv] = useState(MULTIVARIATE_HOUSING);
  const [mode, setMode] = useState<'ols' | 'ridge'>('ols');
  const [alpha, setAlpha] = useState(1.0);
  const [standardize, setStandardize] = useState(true);
  const [queryValues, setQueryValues] = useState<number[]>([]);
  const [hydrated, setHydrated] = useState(false);

  const fitSvgRef = useRef<SVGSVGElement | null>(null);
  const residSvgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const restored = readUrlToolState<RegressionUrlState>();
    if (restored) {
      if (typeof restored.csv === 'string' && restored.csv.trim().length > 0) setCsv(restored.csv);
      if (restored.mode === 'ols' || restored.mode === 'ridge') setMode(restored.mode);
      if (typeof restored.alpha === 'number' && Number.isFinite(restored.alpha) && restored.alpha >= 0) {
        setAlpha(restored.alpha);
      }
      if (typeof restored.standardize === 'boolean') setStandardize(restored.standardize);
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeUrlToolState({
      csv,
      mode,
      alpha: Number(alpha.toFixed(4)),
      standardize,
    });
  }, [csv, mode, alpha, standardize, hydrated]);

  const result = useMemo(() => {
    try {
      const table = parseNumericTable(csv);
      if (table.headers.length < 2) {
        return { error: 'Provide at least 2 columns: one or more numeric features followed by the numeric target y.' };
      }
      const nFeatures = table.headers.length - 1;
      const featureNames = table.headers.slice(0, nFeatures);
      const targetName = table.headers[nFeatures];
      const X = table.rows.map((row) => row.slice(0, nFeatures));
      const y = table.rows.map((row) => row[nFeatures]);

      if (X.length < 3) {
        return { error: 'Provide at least 3 rows so the model can fit coefficients and residuals.' };
      }
      if (mode === 'ols' && X.length <= nFeatures) {
        return {
          error: `OLS requires more rows (${X.length}) than features (${nFeatures}). Add rows or switch to Ridge Regression (α > 0).`,
        };
      }

      const { scaled, means, stds } = standardizeMatrix(X);
      const useScaled = mode === 'ridge' && standardize;
      const XFit = useScaled ? scaled : X;

      let rawCoef: number[];
      let rawIntercept: number;
      let predictions: number[];

      if (mode === 'ols') {
        const model = new Linear.LinearRegression();
        model.fit(X, y);
        rawCoef = model.coef;
        rawIntercept = model.predict([new Array(nFeatures).fill(0)])[0];
        predictions = model.predict(X);
      } else {
        const model = new Linear.RidgeRegression({ alpha, fitIntercept: true });
        model.fit(XFit, y);
        const fittedCoef = model.coef;
        const fittedIntercept = model.predict([new Array(nFeatures).fill(0)])[0];
        predictions = model.predict(XFit);
        if (useScaled) {
          const unscaled = unscaleLinearWeights(fittedCoef, fittedIntercept, means, stds);
          rawCoef = unscaled.coef;
          rawIntercept = unscaled.intercept;
        } else {
          rawCoef = fittedCoef;
          rawIntercept = fittedIntercept;
        }
      }

      const yMean = y.reduce((s, v) => s + v, 0) / y.length;
      const yStd = Math.sqrt(y.reduce((s, v) => s + (v - yMean) ** 2, 0) / y.length) || 1;
      const standardizedBetas = rawCoef.map((c, j) => (c * stds[j]) / yStd);
      const maxAbsBeta = Math.max(...standardizedBetas.map((b) => Math.abs(b)), 1e-9);
      const metrics = computeRegressionMetrics(y, predictions, nFeatures);

      return {
        error: '',
        featureNames,
        targetName,
        X,
        y,
        means,
        stds,
        rawCoef,
        rawIntercept,
        standardizedBetas,
        maxAbsBeta,
        predictions,
        metrics,
        isSimple1D: nFeatures === 1,
      };
    } catch (caught) {
      return { error: caught instanceof Error ? caught.message : 'Unable to fit regression model.' };
    }
  }, [csv, mode, alpha, standardize]);

  // Keep prediction query inputs synced to feature count & column means
  useEffect(() => {
    if (!result || result.error || !result.means) return;
    setQueryValues((prev) =>
      prev.length === result.means!.length
        ? prev
        : result.means!.map((m) => Number(m.toFixed(2))),
    );
  }, [result]);

  const predictedQuery = useMemo(() => {
    if (!result || result.error || !result.rawCoef || queryValues.length !== result.rawCoef.length) {
      return null;
    }
    let yHat = result.rawIntercept!;
    for (let j = 0; j < result.rawCoef.length; j++) {
      yHat += result.rawCoef[j] * (Number.isFinite(queryValues[j]) ? queryValues[j] : 0);
    }
    return yHat;
  }, [result, queryValues]);

  const exportCsv = () => {
    if (!result || result.error || !result.X) return;
    const header = [...result.featureNames!, result.targetName!, 'predicted', 'residual'].join(',');
    const rows = result.X.map((row, i) =>
      [...row, result.y![i], result.predictions![i].toFixed(6), result.metrics!.residuals[i].toFixed(6)].join(','),
    );
    const meta = [
      `# model=${mode === 'ols' ? 'LinearRegression' : 'RidgeRegression'}`,
      ...(mode === 'ridge' ? [`# alpha=${alpha}`, `# standardized=${standardize}`] : []),
      `# r2=${result.metrics!.r2.toFixed(6)}, adjusted_r2=${result.metrics!.adjustedR2.toFixed(6)}, rmse=${result.metrics!.rmse.toFixed(6)}, mae=${result.metrics!.mae.toFixed(6)}`,
      `# intercept=${result.rawIntercept!.toFixed(6)}`,
    ];
    downloadText('linear-regression-results.csv', [...meta, header, ...rows].join('\n'));
  };

  const jsCode = result && !result.error && result.X
    ? mode === 'ols'
      ? `import { Linear, Metrics } from '@kanaries/ml';

const X = ${JSON.stringify(result.X)};
const y = ${JSON.stringify(result.y)};

const model = new Linear.LinearRegression();
model.fit(X, y);
const yPred = model.predict(X);

console.log({
  coef: model.coef,
  intercept: model.predict([${JSON.stringify(new Array(result.featureNames!.length).fill(0))}])[0],
  r2: Metrics.r2Score(yPred, y),
  rmse: Math.sqrt(Metrics.meanSquaredError(yPred, y)),
});`
      : `import { Linear, Pipeline, Metrics, utils } from '@kanaries/ml';

const X = ${JSON.stringify(result.X)};
const y = ${JSON.stringify(result.y)};

${standardize
  ? `const model = new Pipeline({
  steps: [
    ['scaler', new utils.Preprocessing.StandardScaler()],
    ['ridge', new Linear.RidgeRegression({ alpha: ${alpha}, fitIntercept: true })],
  ],
});`
  : `const model = new Linear.RidgeRegression({ alpha: ${alpha}, fitIntercept: true });`}
model.fit(X, y);
const yPred = model.predict(X);

console.log({
  r2: Metrics.r2Score(yPred, y),
  rmse: Math.sqrt(Metrics.meanSquaredError(yPred, y)),
  predictions: yPred,
});`
    : '';

  const pyCode = result && !result.error && result.X
    ? mode === 'ols'
      ? `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

X = np.array(${JSON.stringify(result.X)}, dtype=float)
y = np.array(${JSON.stringify(result.y)}, dtype=float)

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
print({
    "coef": model.coef_.tolist(),
    "intercept": float(model.intercept_),
    "r2": float(r2_score(y, y_pred)),
    "rmse": float(root_mean_squared_error(y, y_pred)),
})`
      : `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

X = np.array(${JSON.stringify(result.X)}, dtype=float)
y = np.array(${JSON.stringify(result.y)}, dtype=float)

model = ${standardize ? `make_pipeline(StandardScaler(), Ridge(alpha=${alpha}))` : `Ridge(alpha=${alpha})`}.fit(X, y)
y_pred = model.predict(X)
print({
    "r2": float(r2_score(y, y_pred)),
    "rmse": float(root_mean_squared_error(y, y_pred)),
})`
    : '';

  // SVG chart dimensions
  const W = 340;
  const H = 250;
  const pad = { top: 18, right: 18, bottom: 38, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const fitPlot = useMemo(() => {
    if (!result || result.error || !result.X) return null;
    if (result.isSimple1D) {
      const xs = result.X.map((r) => r[0]);
      const ys = result.y!;
      const [xMin, xMax] = niceRange(xs);
      const [yMin, yMax] = niceRange([...ys, ...result.predictions!]);
      const sx = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
      const sy = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;
      const lineY0 = result.rawIntercept! + result.rawCoef![0] * xMin;
      const lineY1 = result.rawIntercept! + result.rawCoef![0] * xMax;
      return {
        kind: '1d' as const,
        xLabel: result.featureNames![0],
        yLabel: result.targetName!,
        xMin,
        xMax,
        yMin,
        yMax,
        points: xs.map((x, i) => ({
          cx: sx(x),
          cy: sy(ys[i]),
          py: sy(result.predictions![i]),
        })),
        line: { x1: sx(xMin), y1: sy(lineY0), x2: sx(xMax), y2: sy(lineY1) },
      };
    }
    const combined = [...result.y!, ...result.predictions!];
    const [lo, hi] = niceRange(combined);
    const sx = (v: number) => pad.left + ((v - lo) / (hi - lo)) * plotW;
    const sy = (v: number) => pad.top + (1 - (v - lo) / (hi - lo)) * plotH;
    return {
      kind: 'actual-vs-pred' as const,
      xLabel: `Actual ${result.targetName}`,
      yLabel: `Predicted ${result.targetName}`,
      xMin: lo,
      xMax: hi,
      yMin: lo,
      yMax: hi,
      points: result.y!.map((yv, i) => ({
        cx: sx(yv),
        cy: sy(result.predictions![i]),
        py: sy(yv),
      })),
      line: { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi) },
    };
  }, [result, plotW, plotH]);

  const residPlot = useMemo(() => {
    if (!result || result.error || !result.predictions || !result.metrics) return null;
    const [xMin, xMax] = niceRange(result.predictions);
    const maxAbsR = Math.max(...result.metrics.residuals.map((r) => Math.abs(r)), 1e-6) * 1.15;
    const sx = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const sy = (v: number) => pad.top + (1 - (v + maxAbsR) / (2 * maxAbsR)) * plotH;
    return {
      xMin,
      xMax,
      maxAbsR,
      zeroY: sy(0),
      points: result.predictions.map((p, i) => ({
        cx: sx(p),
        cy: sy(result.metrics!.residuals[i]),
      })),
    };
  }, [result, plotW, plotH]);

  return (
    <div className={styles.calculator}>
      <div className={styles.grid}>
        <div className={styles.panel}>
          <h2 className={styles.panelTitle}>1. Paste CSV features &amp; target column</h2>
          <DataInput
            label="feature_1, ..., target (CSV or TSV)"
            hint="All columns except the last are predictors (X). The last column is the numeric target (y)."
            value={csv}
            onChange={setCsv}
            examples={[
              { label: 'Multivariate Housing', value: MULTIVARIATE_HOUSING },
              { label: '1D Ad Spend', value: SIMPLE_AD_SPEND },
              { label: 'Collinear (Ridge Demo)', value: COLLINEAR_DEMO },
            ]}
          />

          <div className={styles.sliderGrid}>
            <div className={styles.inputLabel}>2. Estimator &amp; L2 regularization</div>
            <div className={styles.tabs} role="tablist" aria-label="Regression estimator">
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'ols'}
                className={`${styles.tab} ${mode === 'ols' ? styles.tabActive : ''}`}
                onClick={() => setMode('ols')}
              >
                OLS Linear Regression
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'ridge'}
                className={`${styles.tab} ${mode === 'ridge' ? styles.tabActive : ''}`}
                onClick={() => setMode('ridge')}
              >
                Ridge Regression (L2)
              </button>
            </div>

            {mode === 'ridge' && (
              <>
                <div className={styles.sliderRow}>
                  <span>Alpha (α)</span>
                  <input
                    type="range"
                    aria-label="Ridge L2 regularization alpha"
                    min={-2}
                    max={2}
                    step={0.05}
                    value={Math.log10(Math.max(alpha, 0.01))}
                    onChange={(e) => setAlpha(Number((10 ** Number(e.target.value)).toFixed(3)))}
                  />
                  <strong>{formatMetric(alpha, 3)}</strong>
                </div>
                <div className={styles.sampleActions}>
                  {[0.01, 0.1, 1, 10, 50].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      className={styles.sampleButton}
                      onClick={() => setAlpha(preset)}
                    >
                      α = {preset}
                    </button>
                  ))}
                </div>
                <label className={styles.panelHint} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={standardize}
                    onChange={(e) => setStandardize(e.target.checked)}
                  />
                  Standardize predictors before L2 penalty (recommended)
                </label>
              </>
            )}
          </div>
        </div>

        <div className={styles.panel}>
          <h2 className={styles.panelTitle}>Fit summary, coefficients &amp; residual diagnostics</h2>
          {result.error ? (
            <div className={styles.error}>{result.error}</div>
          ) : (
            <>
              <StatCards
                items={[
                  { label: 'R² Score', value: result.metrics!.r2, digits: 4 },
                  { label: 'Adjusted R²', value: result.metrics!.adjustedR2, digits: 4 },
                  { label: 'RMSE', value: result.metrics!.rmse, digits: 4 },
                  { label: 'MAE', value: result.metrics!.mae, digits: 4 },
                  { label: 'Intercept (β₀)', value: result.rawIntercept!, digits: 4 },
                  { label: 'Samples (n)', value: result.X!.length, digits: 0 },
                  { label: 'Features (p)', value: result.featureNames!.length, digits: 0 },
                  { label: 'L2 Penalty α', value: mode === 'ols' ? 0 : alpha, digits: 3 },
                ]}
              />

              <div className={styles.visualGrid}>
                {fitPlot && (
                  <div className={styles.chartCard}>
                    <h3 className={styles.chartTitle}>
                      {fitPlot.kind === '1d' ? 'Fitted Regression Line' : 'Actual vs Predicted'}
                    </h3>
                    <svg
                      ref={fitSvgRef}
                      className={styles.chartSvg}
                      viewBox={`0 0 ${W} ${H}`}
                      role="img"
                      aria-label={fitPlot.kind === '1d' ? 'Fitted linear regression line' : 'Actual versus predicted scatter plot'}
                    >
                      <line x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} stroke="currentColor" strokeOpacity="0.25" />
                      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} stroke="currentColor" strokeOpacity="0.25" />
                      <line
                        x1={fitPlot.line.x1}
                        y1={fitPlot.line.y1}
                        x2={fitPlot.line.x2}
                        y2={fitPlot.line.y2}
                        stroke={fitPlot.kind === '1d' ? '#1d3f72' : '#8a8275'}
                        strokeWidth={fitPlot.kind === '1d' ? '2.4' : '1.4'}
                        strokeDasharray={fitPlot.kind === '1d' ? undefined : '4 4'}
                      />
                      {fitPlot.points.map((pt, idx) => (
                        <g key={idx}>
                          {fitPlot.kind === '1d' && (
                            <line
                              x1={pt.cx}
                              y1={pt.cy}
                              x2={pt.cx}
                              y2={pt.py}
                              stroke="#d9534f"
                              strokeOpacity="0.45"
                              strokeWidth="1.2"
                            />
                          )}
                          <circle cx={pt.cx} cy={pt.cy} r="4.2" fill="#1d3f72" fillOpacity="0.88" />
                        </g>
                      ))}
                      <text className={styles.axisLabel} x={pad.left} y={H - 22}>{formatMetric(fitPlot.xMin, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left + plotW} y={H - 22} textAnchor="end">{formatMetric(fitPlot.xMax, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left - 6} y={pad.top + plotH} textAnchor="end">{formatMetric(fitPlot.yMin, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left - 6} y={pad.top + 8} textAnchor="end">{formatMetric(fitPlot.yMax, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left + plotW / 2} y={H - 5} textAnchor="middle">
                        {fitPlot.xLabel}
                      </text>
                      <text
                        className={styles.axisLabel}
                        x={12}
                        y={pad.top + plotH / 2}
                        textAnchor="middle"
                        transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
                      >
                        {fitPlot.yLabel}
                      </text>
                    </svg>
                  </div>
                )}

                {residPlot && (
                  <div className={styles.chartCard}>
                    <h3 className={styles.chartTitle}>Residuals (y − ŷ) vs Predicted</h3>
                    <svg
                      ref={residSvgRef}
                      className={styles.chartSvg}
                      viewBox={`0 0 ${W} ${H}`}
                      role="img"
                      aria-label="Residuals versus predicted values plot"
                    >
                      <line x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} stroke="currentColor" strokeOpacity="0.25" />
                      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} stroke="currentColor" strokeOpacity="0.25" />
                      <line
                        x1={pad.left}
                        y1={residPlot.zeroY}
                        x2={pad.left + plotW}
                        y2={residPlot.zeroY}
                        stroke="#2f9e63"
                        strokeDasharray="4 4"
                        strokeWidth="1.5"
                      />
                      {residPlot.points.map((pt, idx) => (
                        <g key={idx}>
                          <line x1={pt.cx} y1={residPlot.zeroY} x2={pt.cx} y2={pt.cy} stroke="#d9534f" strokeOpacity="0.38" strokeWidth="1.2" />
                          <circle cx={pt.cx} cy={pt.cy} r="4.2" fill="#d9534f" fillOpacity="0.88" />
                        </g>
                      ))}
                      <text className={styles.axisLabel} x={pad.left} y={H - 22}>{formatMetric(residPlot.xMin, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left + plotW} y={H - 22} textAnchor="end">{formatMetric(residPlot.xMax, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left - 6} y={pad.top + 8} textAnchor="end">+{formatMetric(residPlot.maxAbsR, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left - 6} y={residPlot.zeroY + 3} textAnchor="end">0</text>
                      <text className={styles.axisLabel} x={pad.left - 6} y={pad.top + plotH} textAnchor="end">−{formatMetric(residPlot.maxAbsR, 1)}</text>
                      <text className={styles.axisLabel} x={pad.left + plotW / 2} y={H - 5} textAnchor="middle">
                        Predicted {result.targetName}
                      </text>
                      <text
                        className={styles.axisLabel}
                        x={12}
                        y={pad.top + plotH / 2}
                        textAnchor="middle"
                        transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
                      >
                        Residual
                      </text>
                    </svg>
                  </div>
                )}
              </div>

              <div className={styles.modelTable}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Coefficient (β)</th>
                      <th>Standardized β</th>
                      <th>Relative Magnitude</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>(Intercept)</td>
                      <td>{formatMetric(result.rawIntercept!, 4)}</td>
                      <td>—</td>
                      <td>—</td>
                    </tr>
                    {result.featureNames!.map((name, idx) => {
                      const pct = Math.min(100, (Math.abs(result.standardizedBetas![idx]) / result.maxAbsBeta!) * 100);
                      return (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{formatMetric(result.rawCoef![idx], 4)}</td>
                          <td>{formatMetric(result.standardizedBetas![idx], 4)}</td>
                          <td>
                            <div className={styles.barTrack}>
                              <div className={styles.barFill} style={{ width: `${pct.toFixed(1)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className={styles.probability}>
                <div className={styles.inputLabel}>Predict a new observation</div>
                <div className={styles.matrixInputs} style={{ marginTop: '0.5rem' }}>
                  {result.featureNames!.map((name, idx) => (
                    <div className={styles.field} key={name}>
                      <label htmlFor={`pred-${name}`}>{name}</label>
                      <input
                        id={`pred-${name}`}
                        type="number"
                        step="any"
                        className={styles.input}
                        value={queryValues[idx] ?? 0}
                        onChange={(e) => {
                          const next = queryValues.slice();
                          next[idx] = Number(e.target.value);
                          setQueryValues(next);
                        }}
                      />
                    </div>
                  ))}
                </div>
                {predictedQuery !== null && (
                  <div style={{ marginTop: '0.7rem', fontSize: '0.85rem' }}>
                    Predicted {result.targetName}: <strong>{formatMetric(predictedQuery, 4)}</strong>
                  </div>
                )}
              </div>

              <div className={styles.actions}>
                <button type="button" className={styles.button} onClick={exportCsv}>
                  Download Predictions CSV
                </button>
                <button
                  type="button"
                  className={styles.buttonSecondary}
                  onClick={() => downloadSvgAsPng(fitSvgRef.current, 'linear-regression-fit.png')}
                >
                  Download Fit PNG
                </button>
                <button
                  type="button"
                  className={styles.buttonSecondary}
                  onClick={() => downloadSvgAsPng(residSvgRef.current, 'linear-regression-residuals.png')}
                >
                  Download Residuals PNG
                </button>
              </div>

              <CodeTabs javascript={jsCode} python={pyCode} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
