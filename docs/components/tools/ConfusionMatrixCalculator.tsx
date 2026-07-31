'use client';

import { useDeferredValue, useMemo, useRef, useState } from 'react';
import { Metrics } from '@kanaries/ml';
import { CodeTabs } from './CodeTabs';
import { DataInput, parseNumericTable } from './DataInput';
import { MatrixHeatmap } from './MatrixHeatmap';
import { StatCards, formatMetric } from './StatCards';
import { downloadSvgAsPng, downloadText } from './clientUtils';
import { summarizeConfusionMatrix } from './calculatorMetrics';
import styles from './interactive.module.css';

const MULTICLASS_EXAMPLE = `y_true,y_pred
0,0
0,0
0,1
1,1
1,1
1,2
2,2
2,2
2,1
2,2`;

const BINARY_EXAMPLE = `y_true,y_pred
1,1
1,1
1,0
1,1
0,0
0,1
0,0
0,0`;

type Calculation = {
  labels: number[];
  matrix: number[][];
  yTrue: number[];
  yPred: number[];
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  macroF1: number;
  microF1: number;
  weightedF1: number;
  mcc: number;
  kappa: number;
  perClass: Array<{ label: number; precision: number; recall: number; f1: number; support: number }>;
};

function expandMatrix(tn: number, fp: number, fn: number, tp: number) {
  const yTrue: number[] = [];
  const yPred: number[] = [];
  const add = (truth: number, prediction: number, count: number) => {
    for (let index = 0; index < count; index++) {
      yTrue.push(truth);
      yPred.push(prediction);
    }
  };
  add(0, 0, tn);
  add(0, 1, fp);
  add(1, 0, fn);
  add(1, 1, tp);
  return { yTrue, yPred };
}

export function ConfusionMatrixCalculator() {
  const [mode, setMode] = useState<'matrix' | 'pairs'>('matrix');
  const [counts, setCounts] = useState({ tp: 42, fp: 6, fn: 9, tn: 63 });
  const [pairs, setPairs] = useState(MULTICLASS_EXAMPLE);
  const deferredCounts = useDeferredValue(counts);
  const deferredPairs = useDeferredValue(pairs);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const result = useMemo<{ calculation: Calculation | null; error: string }>(() => {
    try {
      let yTrue: number[];
      let yPred: number[];
      if (mode === 'matrix') {
        const values = Object.values(deferredCounts);
        if (values.some((value) => !Number.isInteger(value) || value < 0)) {
          throw new Error('Matrix counts must be non-negative whole numbers.');
        }
        const total = values.reduce((sum, value) => sum + value, 0);
        if (total === 0) throw new Error('At least one matrix cell must be greater than zero.');
        if (total > 20000) throw new Error('For browser performance, keep the total count at or below 20,000.');
        ({ yTrue, yPred } = expandMatrix(deferredCounts.tn, deferredCounts.fp, deferredCounts.fn, deferredCounts.tp));
      } else {
        const parsed = parseNumericTable(deferredPairs);
        if (parsed.headers.length !== 2) throw new Error('Use exactly two columns: y_true and y_pred.');
        yTrue = parsed.rows.map((row) => row[0]);
        yPred = parsed.rows.map((row) => row[1]);
      }

      const labels = Array.from(new Set([...yTrue, ...yPred])).sort((a, b) => a - b);
      if (labels.length < 2) throw new Error('The data must contain at least two classes.');
      const matrix = Metrics.confusionMatrix(yPred, yTrue, labels);
      const summary = summarizeConfusionMatrix(matrix, labels);
      const primary = labels.length === 2
        ? {
            precision: Metrics.precisionScore(yPred, yTrue, { positiveLabel: labels[1] }),
            recall: Metrics.recallScore(yPred, yTrue, { positiveLabel: labels[1] }),
            f1: Metrics.f1Score(yPred, yTrue, { positiveLabel: labels[1] }),
          }
        : {
            precision: Metrics.precisionScore(yPred, yTrue, { average: 'macro' }),
            recall: Metrics.recallScore(yPred, yTrue, { average: 'macro' }),
            f1: Metrics.f1Score(yPred, yTrue, { average: 'macro' }),
          };
      const calculation: Calculation = {
        labels,
        matrix,
        yTrue,
        yPred,
        accuracy: Metrics.accuracyScore(yPred, yTrue),
        precision: primary.precision,
        recall: primary.recall,
        specificity: summary.macroSpecificity,
        f1: primary.f1,
        macroF1: summary.macroF1,
        microF1: summary.accuracy,
        weightedF1: summary.weightedF1,
        mcc: summary.mcc,
        kappa: summary.kappa,
        perClass: summary.perClass,
      };
      return { calculation, error: '' };
    } catch (caught) {
      return {
        calculation: null,
        error: caught instanceof Error ? caught.message : 'The metrics could not be calculated.',
      };
    }
  }, [deferredCounts, deferredPairs, mode]);

  const calculation = result.calculation;
  const jsCode = calculation ? `import { Metrics } from '@kanaries/ml';

const yTrue = ${JSON.stringify(calculation.yTrue)};
const yPred = ${JSON.stringify(calculation.yPred)};

const labels = ${JSON.stringify(calculation.labels)};
const matrix = Metrics.confusionMatrix(yPred, yTrue, labels);
const accuracy = Metrics.accuracyScore(yPred, yTrue);
const precision = Metrics.precisionScore(yPred, yTrue, { average: 'macro' });
const recall = Metrics.recallScore(yPred, yTrue, { average: 'macro' });
const f1 = Metrics.f1Score(yPred, yTrue, { average: 'macro' });

console.log({ matrix, accuracy, precision, recall, f1 });` : '';

  const pythonCode = calculation ? `from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)

y_true = ${JSON.stringify(calculation.yTrue)}
y_pred = ${JSON.stringify(calculation.yPred)}
labels = ${JSON.stringify(calculation.labels)}

print(confusion_matrix(y_true, y_pred, labels=labels))
print(classification_report(y_true, y_pred, output_dict=True))
print(matthews_corrcoef(y_true, y_pred))
print(cohen_kappa_score(y_true, y_pred))` : '';

  const exportCsv = () => {
    if (!calculation) return;
    const header = ['true\\predicted', ...calculation.labels].join(',');
    const rows = calculation.matrix.map((row, index) => [calculation.labels[index], ...row].join(','));
    const metrics = [
      '',
      'metric,value',
      `accuracy,${calculation.accuracy}`,
      `macro_f1,${calculation.macroF1}`,
      `micro_f1,${calculation.microF1}`,
      `weighted_f1,${calculation.weightedF1}`,
      `mcc,${calculation.mcc}`,
      `cohen_kappa,${calculation.kappa}`,
    ];
    downloadText('confusion-matrix-metrics.csv', [header, ...rows, ...metrics].join('\n'));
  };

  return (
    <div className={styles.calculator}>
      <div className={styles.tabs} role="tablist" aria-label="Input mode">
        <button
          className={`${styles.tab} ${mode === 'matrix' ? styles.tabActive : ''}`}
          onClick={() => setMode('matrix')}
          role="tab"
          aria-selected={mode === 'matrix'}
          type="button"
        >
          Enter a 2×2 matrix
        </button>
        <button
          className={`${styles.tab} ${mode === 'pairs' ? styles.tabActive : ''}`}
          onClick={() => setMode('pairs')}
          role="tab"
          aria-selected={mode === 'pairs'}
          type="button"
        >
          Paste y_true / y_pred
        </button>
      </div>

      <div className={styles.grid}>
        <section className={styles.panel}>
          <h2 className={styles.panelTitle}>Input</h2>
          {mode === 'matrix' ? (
            <>
              <p className={styles.panelHint}>Enter binary-classification counts. Every metric updates immediately.</p>
              <div className={styles.matrixInputs}>
                {(['tp', 'fp', 'fn', 'tn'] as const).map((key) => (
                  <div className={styles.field} key={key}>
                    <label htmlFor={`matrix-${key}`}>{key.toUpperCase()} — {{ tp: 'true positive', fp: 'false positive', fn: 'false negative', tn: 'true negative' }[key]}</label>
                    <input
                      className={styles.input}
                      id={`matrix-${key}`}
                      min={0}
                      step={1}
                      type="number"
                      value={counts[key]}
                      onChange={(event) => setCounts((current) => ({ ...current, [key]: Number(event.target.value) }))}
                    />
                  </div>
                ))}
              </div>
            </>
          ) : (
            <DataInput
              value={pairs}
              onChange={setPairs}
              label="True and predicted labels"
              hint="The first column is y_true; the second is y_pred. Numeric class labels are supported, including multiclass data."
              examples={[
                { label: 'multiclass data', value: MULTICLASS_EXAMPLE },
                { label: 'binary data', value: BINARY_EXAMPLE },
              ]}
            />
          )}
          {result.error && <div className={styles.error}>{result.error}</div>}
        </section>

        <section className={styles.panel} aria-live="polite">
          <h2 className={styles.panelTitle}>Results</h2>
          <p className={styles.panelHint}>
            {calculation && calculation.labels.length > 2
              ? `Multiclass report across ${calculation.labels.length} labels; headline precision, recall, and F1 are macro averages.`
              : 'For binary data, precision, recall, and F1 use the greater numeric label as the positive class.'}
          </p>
          {calculation && (
            <>
              <StatCards items={[
                { label: 'Accuracy', value: calculation.accuracy },
                { label: calculation.labels.length > 2 ? 'Precision (macro)' : 'Precision', value: calculation.precision },
                { label: calculation.labels.length > 2 ? 'Recall (macro)' : 'Recall', value: calculation.recall },
                { label: 'Specificity (macro)', value: calculation.specificity },
                { label: calculation.labels.length > 2 ? 'F1 (macro)' : 'F1 score', value: calculation.f1 },
                { label: 'Micro F1', value: calculation.microF1 },
                { label: 'Weighted F1', value: calculation.weightedF1 },
                { label: 'MCC', value: calculation.mcc },
                { label: "Cohen's kappa", value: calculation.kappa },
              ]} />

              <div className={styles.visualGrid}>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>Confusion matrix heatmap</h3>
                  <MatrixHeatmap matrix={calculation.matrix} labels={calculation.labels} svgRef={svgRef} />
                </div>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>Per-class performance</h3>
                  <div className={styles.barList}>
                    {calculation.perClass.flatMap((row) => ([
                      ['P', row.precision],
                      ['R', row.recall],
                      ['F1', row.f1],
                    ] as const).map(([metric, value]) => (
                      <div className={styles.barRow} key={`${row.label}-${metric}`}>
                        <span>Class {row.label} {metric}</span>
                        <span className={styles.barTrack}><span className={styles.barFill} style={{ width: `${value * 100}%`, display: 'block' }} /></span>
                        <span>{formatMetric(value)}</span>
                      </div>
                    )))}
                  </div>
                </div>
              </div>

              <div className={styles.actions}>
                <button className={styles.button} onClick={exportCsv} type="button">Download CSV</button>
                <button className={styles.buttonSecondary} onClick={() => downloadSvgAsPng(svgRef.current, 'confusion-matrix.png')} type="button">Download PNG</button>
              </div>
              <CodeTabs javascript={jsCode} python={pythonCode} />
            </>
          )}
        </section>
      </div>
    </div>
  );
}
