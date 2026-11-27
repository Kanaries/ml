'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Metrics } from '@kanaries/ml';
import styles from './interactive.module.css';
import { DataInput, parseNumericTable } from './DataInput';
import { StatCards, formatMetric } from './StatCards';
import { CodeTabs } from './CodeTabs';
import { downloadSvgAsPng, downloadText, readUrlToolState, writeUrlToolState } from './clientUtils';
import {
  brierScore,
  computeAveragePrecision,
  evaluateThreshold,
  findOptimalThresholds,
} from './calculatorMetrics';

const WELL_SEPARATED_SAMPLE = `y_true,y_score
1,0.96
1,0.91
1,0.87
1,0.82
0,0.78
1,0.74
1,0.69
0,0.63
1,0.58
1,0.52
0,0.47
1,0.41
0,0.36
0,0.31
0,0.26
1,0.21
0,0.17
0,0.12
0,0.08
0,0.04`;

const IMBALANCED_FRAUD_SAMPLE = `y_true,y_score
1,0.93
1,0.84
0,0.79
0,0.71
1,0.66
0,0.59
0,0.51
0,0.44
0,0.38
0,0.34
0,0.29
0,0.25
0,0.21
0,0.18
0,0.15
0,0.12
0,0.09
0,0.07
0,0.05
0,0.03
0,0.02
0,0.01`;

const NOISY_OVERLAP_SAMPLE = `y_true,y_score
1,0.88
0,0.82
1,0.76
0,0.71
1,0.64
1,0.59
0,0.55
1,0.51
0,0.48
1,0.44
0,0.39
1,0.33
0,0.28
0,0.21
0,0.15
0,0.09`;

type RocUrlState = {
  csv?: string;
  threshold?: number;
};

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export function RocAucCalculator() {
  const [csv, setCsv] = useState(WELL_SEPARATED_SAMPLE);
  const [threshold, setThreshold] = useState(0.5);
  const [hydrated, setHydrated] = useState(false);
  const rocSvgRef = useRef<SVGSVGElement | null>(null);
  const prSvgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const restored = readUrlToolState<RocUrlState>();
    if (restored) {
      if (typeof restored.csv === 'string' && restored.csv.trim().length > 0) setCsv(restored.csv);
      if (typeof restored.threshold === 'number' && Number.isFinite(restored.threshold)) {
        setThreshold(restored.threshold);
      }
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeUrlToolState({ csv, threshold: Number(threshold.toFixed(4)) });
  }, [csv, threshold, hydrated]);

  const analysis = useMemo(() => {
    try {
      const table = parseNumericTable(csv);
      if (table.headers.length < 2) {
        return { error: 'Provide at least two columns: binary y_true (0/1) and continuous y_score.' };
      }
      const yTrue = table.rows.map((row) => row[0]);
      const yScores = table.rows.map((row) => row[1]);
      const classes = Array.from(new Set(yTrue)).sort((a, b) => a - b);
      if (classes.length !== 2) {
        return { error: `ROC-AUC requires exactly 2 unique classes in column 1 (found ${classes.length}).` };
      }
      const positiveLabel = classes[1];
      const roc = Metrics.rocCurve(yTrue, yScores, positiveLabel);
      const rocAuc = Metrics.rocAucScore(yTrue, yScores, positiveLabel);
      const pr = Metrics.precisionRecallCurve(yTrue, yScores, positiveLabel);
      const averagePrecision = computeAveragePrecision(pr.precision, pr.recall);
      const minScore = Math.min(...yScores);
      const maxScore = Math.max(...yScores);
      const isProbability = minScore >= 0 && maxScore <= 1;
      const brier = isProbability ? brierScore(yTrue, yScores, positiveLabel) : null;
      const prevalence = yTrue.filter((v) => v === positiveLabel).length / yTrue.length;
      const clampedThreshold = Math.min(maxScore, Math.max(minScore, threshold));
      const atThreshold = evaluateThreshold(yTrue, yScores, clampedThreshold, positiveLabel);
      const optimal = findOptimalThresholds(yTrue, yScores, positiveLabel);

      return {
        error: '',
        yTrue,
        yScores,
        positiveLabel,
        roc,
        rocAuc,
        pr,
        averagePrecision,
        brier,
        prevalence,
        minScore,
        maxScore,
        clampedThreshold,
        atThreshold,
        optimal,
      };
    } catch (caught) {
      return { error: caught instanceof Error ? caught.message : 'Unable to parse ROC/PR input.' };
    }
  }, [csv, threshold]);

  const exportCsv = () => {
    if (!analysis || analysis.error || !analysis.roc) return;
    const rows = [
      'metric,value',
      `roc_auc,${analysis.rocAuc.toFixed(6)}`,
      `average_precision,${analysis.averagePrecision.toFixed(6)}`,
      `prevalence,${analysis.prevalence.toFixed(6)}`,
      ...(analysis.brier !== null ? [`brier_score,${analysis.brier.toFixed(6)}`] : []),
      `operating_threshold,${analysis.clampedThreshold.toFixed(4)}`,
      `f1_at_threshold,${analysis.atThreshold.f1.toFixed(6)}`,
      `precision_at_threshold,${analysis.atThreshold.precision.toFixed(6)}`,
      `recall_tpr_at_threshold,${analysis.atThreshold.recall.toFixed(6)}`,
      `fpr_at_threshold,${analysis.atThreshold.fpr.toFixed(6)}`,
      `youden_j_optimal_threshold,${analysis.optimal.bestYouden.threshold.toFixed(4)}`,
      `best_f1_threshold,${analysis.optimal.bestF1.threshold.toFixed(4)}`,
      '',
      'roc_threshold,fpr,tpr',
      ...analysis.roc.fpr.map((f, i) => `${analysis.roc.thresholds[i]},${f.toFixed(6)},${analysis.roc.tpr[i].toFixed(6)}`),
    ];
    downloadText('roc-pr-curve-results.csv', rows.join('\n'));
  };

  const jsCode = analysis && !analysis.error && analysis.yTrue
    ? `import { Metrics } from '@kanaries/ml';

const yTrue = ${JSON.stringify(analysis.yTrue)};
const yScores = ${JSON.stringify(analysis.yScores)};

const rocAuc = Metrics.rocAucScore(yTrue, yScores, ${analysis.positiveLabel});
const roc = Metrics.rocCurve(yTrue, yScores, ${analysis.positiveLabel});
const pr = Metrics.precisionRecallCurve(yTrue, yScores, ${analysis.positiveLabel});

const threshold = ${Number(analysis.clampedThreshold.toFixed(4))};
const yPred = yScores.map((s) => (s >= threshold ? ${analysis.positiveLabel} : 0));

console.log({
  rocAuc,
  f1: Metrics.f1Score(yPred, yTrue),
  precision: Metrics.precisionScore(yPred, yTrue),
  recall: Metrics.recallScore(yPred, yTrue),
  rocPoints: roc.fpr.length,
  prPoints: pr.recall.length,
});`
    : '';

  const pyCode = analysis && !analysis.error && analysis.yTrue
    ? `import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

y_true = np.array(${JSON.stringify(analysis.yTrue)})
y_scores = np.array(${JSON.stringify(analysis.yScores)})

roc_auc = roc_auc_score(y_true, y_scores)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=${analysis.positiveLabel})
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores, pos_label=${analysis.positiveLabel})
ap = average_precision_score(y_true, y_scores)

threshold = ${Number(analysis.clampedThreshold.toFixed(4))}
y_pred = (y_scores >= threshold).astype(int)
print({"roc_auc": roc_auc, "ap": ap, "f1": f1_score(y_true, y_pred)})`
    : '';

  // SVG coordinate helper for [0,1] x [0,1] curves
  const W = 340;
  const H = 260;
  const pad = { top: 18, right: 18, bottom: 38, left: 42 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const sx = (x: number) => pad.left + clamp01(x) * plotW;
  const sy = (y: number) => pad.top + (1 - clamp01(y)) * plotH;

  const rocPolyline = analysis && !analysis.error && analysis.roc
    ? analysis.roc.fpr.map((f, i) => `${sx(f).toFixed(1)},${sy(analysis.roc.tpr[i]).toFixed(1)}`).join(' ')
    : '';
  const rocArea = rocPolyline
    ? `${sx(0)},${sy(0)} ${rocPolyline} ${sx(1)},${sy(0)}`
    : '';

  // PR curve in descending-recall order + (recall=0, precision=1) endpoint
  const prPoints = analysis && !analysis.error && analysis.pr
    ? [
        ...analysis.pr.recall.map((r, i) => ({ r, p: analysis.pr.precision[i] })),
        { r: 0, p: 1 },
      ]
    : [];
  const prPolyline = prPoints.map((pt) => `${sx(pt.r).toFixed(1)},${sy(pt.p).toFixed(1)}`).join(' ');
  const prArea = prPoints.length > 0
    ? `${sx(prPoints[0].r).toFixed(1)},${sy(0)} ${prPolyline} ${sx(0)},${sy(0)}`
    : '';

  return (
    <div className={styles.calculator}>
      <div className={styles.grid}>
        <div className={styles.panel}>
          <h2 className={styles.panelTitle}>1. Paste binary labels &amp; predicted scores</h2>
          <DataInput
            label="y_true, y_score (CSV or TSV)"
            hint="Column 1: binary ground truth (0/1). Column 2: probability or continuous decision score."
            value={csv}
            onChange={setCsv}
            examples={[
              { label: 'Well-Separated', value: WELL_SEPARATED_SAMPLE },
              { label: 'Imbalanced Fraud', value: IMBALANCED_FRAUD_SAMPLE },
              { label: 'Noisy Overlap', value: NOISY_OVERLAP_SAMPLE },
            ]}
          />

          {analysis && !analysis.error && analysis.atThreshold && (
            <div className={styles.sliderGrid}>
              <div className={styles.inputLabel}>2. Operating decision threshold</div>
              <div className={styles.sliderRow}>
                <span>Threshold</span>
                <input
                  type="range"
                  aria-label="Classification decision threshold"
                  min={analysis.minScore}
                  max={analysis.maxScore}
                  step={Math.max((analysis.maxScore - analysis.minScore) / 200, 0.005)}
                  value={analysis.clampedThreshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
                <strong>{formatMetric(analysis.clampedThreshold, 3)}</strong>
              </div>
              <div className={styles.sampleActions}>
                <button
                  type="button"
                  className={styles.sampleButton}
                  onClick={() => setThreshold(analysis.optimal.bestYouden.threshold)}
                >
                  Best Youden’s J ({formatMetric(analysis.optimal.bestYouden.threshold, 2)})
                </button>
                <button
                  type="button"
                  className={styles.sampleButton}
                  onClick={() => setThreshold(analysis.optimal.bestF1.threshold)}
                >
                  Best F1 ({formatMetric(analysis.optimal.bestF1.threshold, 2)})
                </button>
                <button
                  type="button"
                  className={styles.sampleButton}
                  onClick={() => setThreshold(0.5)}
                >
                  Reset 0.50
                </button>
              </div>
            </div>
          )}
        </div>

        <div className={styles.panel}>
          <h2 className={styles.panelTitle}>ROC-AUC, Precision–Recall &amp; threshold metrics</h2>
          {analysis.error ? (
            <div className={styles.error}>{analysis.error}</div>
          ) : (
            <>
              <StatCards
                items={[
                  { label: 'ROC-AUC', value: analysis.rocAuc!, digits: 4 },
                  { label: 'Avg Precision (PR)', value: analysis.averagePrecision!, digits: 4 },
                  { label: 'F1 @ Threshold', value: analysis.atThreshold!.f1, digits: 4 },
                  { label: 'Youden’s J', value: analysis.atThreshold!.youdenJ, digits: 4 },
                  { label: 'Recall (TPR)', value: analysis.atThreshold!.recall, digits: 4 },
                  { label: 'Precision (PPV)', value: analysis.atThreshold!.precision, digits: 4 },
                  { label: 'FPR (1 − Spec)', value: analysis.atThreshold!.fpr, digits: 4 },
                  {
                    label: analysis.brier != null ? 'Brier Score' : 'Positive Rate',
                    value: analysis.brier != null ? analysis.brier : analysis.prevalence!,
                    digits: 4,
                  },
                ]}
              />

              <div className={styles.visualGrid}>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>
                    ROC Curve (AUC = {formatMetric(analysis.rocAuc!, 3)})
                  </h3>
                  <svg
                    ref={rocSvgRef}
                    className={styles.chartSvg}
                    viewBox={`0 0 ${W} ${H}`}
                    role="img"
                    aria-label="Receiver Operating Characteristic curve"
                  >
                    {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
                      <g key={tick}>
                        <line x1={sx(tick)} y1={sy(0)} x2={sx(tick)} y2={sy(1)} stroke="currentColor" strokeOpacity="0.1" />
                        <line x1={sx(0)} y1={sy(tick)} x2={sx(1)} y2={sy(tick)} stroke="currentColor" strokeOpacity="0.1" />
                        <text className={styles.axisLabel} x={sx(tick)} y={H - 22} textAnchor="middle">{tick.toFixed(2)}</text>
                        <text className={styles.axisLabel} x={pad.left - 8} y={sy(tick) + 3} textAnchor="end">{tick.toFixed(2)}</text>
                      </g>
                    ))}
                    <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#8a8275" strokeDasharray="4 4" strokeWidth="1.2" />
                    <polygon points={rocArea} fill="#1d3f72" fillOpacity="0.14" />
                    <polyline points={rocPolyline} fill="none" stroke="#1d3f72" strokeWidth="2.4" />
                    <circle
                      cx={sx(analysis.atThreshold!.fpr)}
                      cy={sy(analysis.atThreshold!.recall)}
                      r="5.5"
                      fill="#d9534f"
                      stroke="#fff"
                      strokeWidth="2"
                    />
                    <text className={styles.axisLabel} x={pad.left + plotW / 2} y={H - 5} textAnchor="middle">
                      False Positive Rate (FPR)
                    </text>
                    <text
                      className={styles.axisLabel}
                      x={12}
                      y={pad.top + plotH / 2}
                      textAnchor="middle"
                      transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
                    >
                      True Positive Rate (TPR)
                    </text>
                  </svg>
                </div>

                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>
                    Precision–Recall (AP = {formatMetric(analysis.averagePrecision!, 3)})
                  </h3>
                  <svg
                    ref={prSvgRef}
                    className={styles.chartSvg}
                    viewBox={`0 0 ${W} ${H}`}
                    role="img"
                    aria-label="Precision Recall curve"
                  >
                    {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
                      <g key={tick}>
                        <line x1={sx(tick)} y1={sy(0)} x2={sx(tick)} y2={sy(1)} stroke="currentColor" strokeOpacity="0.1" />
                        <line x1={sx(0)} y1={sy(tick)} x2={sx(1)} y2={sy(tick)} stroke="currentColor" strokeOpacity="0.1" />
                        <text className={styles.axisLabel} x={sx(tick)} y={H - 22} textAnchor="middle">{tick.toFixed(2)}</text>
                        <text className={styles.axisLabel} x={pad.left - 8} y={sy(tick) + 3} textAnchor="end">{tick.toFixed(2)}</text>
                      </g>
                    ))}
                    <line
                      x1={sx(0)}
                      y1={sy(analysis.prevalence!)}
                      x2={sx(1)}
                      y2={sy(analysis.prevalence!)}
                      stroke="#8a8275"
                      strokeDasharray="4 4"
                      strokeWidth="1.2"
                    />
                    <polygon points={prArea} fill="#2f9e63" fillOpacity="0.14" />
                    <polyline points={prPolyline} fill="none" stroke="#2f9e63" strokeWidth="2.4" />
                    <circle
                      cx={sx(analysis.atThreshold!.recall)}
                      cy={sy(analysis.atThreshold!.precision)}
                      r="5.5"
                      fill="#d9534f"
                      stroke="#fff"
                      strokeWidth="2"
                    />
                    <text className={styles.axisLabel} x={pad.left + plotW / 2} y={H - 5} textAnchor="middle">
                      Recall (TPR)
                    </text>
                    <text
                      className={styles.axisLabel}
                      x={12}
                      y={pad.top + plotH / 2}
                      textAnchor="middle"
                      transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
                    >
                      Precision
                    </text>
                  </svg>
                </div>
              </div>

              <div className={styles.modelTable}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Threshold point</th>
                      <th>Threshold</th>
                      <th>TP / FP / FN / TN</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Youden’s J</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Current operating point</td>
                      <td>{formatMetric(analysis.atThreshold!.threshold, 3)}</td>
                      <td>
                        {analysis.atThreshold!.tp} / {analysis.atThreshold!.fp} / {analysis.atThreshold!.fn} / {analysis.atThreshold!.tn}
                      </td>
                      <td>{formatMetric(analysis.atThreshold!.precision, 3)}</td>
                      <td>{formatMetric(analysis.atThreshold!.recall, 3)}</td>
                      <td>{formatMetric(analysis.atThreshold!.f1, 3)}</td>
                      <td>{formatMetric(analysis.atThreshold!.youdenJ, 3)}</td>
                    </tr>
                    <tr>
                      <td>Optimal Youden’s J</td>
                      <td>{formatMetric(analysis.optimal!.bestYouden.threshold, 3)}</td>
                      <td>
                        {analysis.optimal!.bestYouden.tp} / {analysis.optimal!.bestYouden.fp} / {analysis.optimal!.bestYouden.fn} / {analysis.optimal!.bestYouden.tn}
                      </td>
                      <td>{formatMetric(analysis.optimal!.bestYouden.precision, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestYouden.recall, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestYouden.f1, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestYouden.youdenJ, 3)}</td>
                    </tr>
                    <tr>
                      <td>Optimal F1</td>
                      <td>{formatMetric(analysis.optimal!.bestF1.threshold, 3)}</td>
                      <td>
                        {analysis.optimal!.bestF1.tp} / {analysis.optimal!.bestF1.fp} / {analysis.optimal!.bestF1.fn} / {analysis.optimal!.bestF1.tn}
                      </td>
                      <td>{formatMetric(analysis.optimal!.bestF1.precision, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestF1.recall, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestF1.f1, 3)}</td>
                      <td>{formatMetric(analysis.optimal!.bestF1.youdenJ, 3)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className={styles.actions}>
                <button type="button" className={styles.button} onClick={exportCsv}>
                  Download ROC/PR CSV
                </button>
                <button
                  type="button"
                  className={styles.buttonSecondary}
                  onClick={() => downloadSvgAsPng(rocSvgRef.current, 'roc-curve.png')}
                >
                  Download ROC PNG
                </button>
                <button
                  type="button"
                  className={styles.buttonSecondary}
                  onClick={() => downloadSvgAsPng(prSvgRef.current, 'pr-curve.png')}
                >
                  Download PR PNG
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
