'use client';

import { useDeferredValue, useMemo, useRef, useState } from 'react';
import { Linear, Metrics } from '@kanaries/ml';
import { BoundaryChart } from './BoundaryChart';
import { CodeTabs } from './CodeTabs';
import { DataInput, parseNumericTable } from './DataInput';
import { MatrixHeatmap } from './MatrixHeatmap';
import { StatCards, formatMetric } from './StatCards';
import { downloadSvgAsPng, downloadText } from './clientUtils';
import { binaryLogLoss, summarizeConfusionMatrix } from './calculatorMetrics';
import styles from './interactive.module.css';

const ONE_FEATURE = `study_hours,passed
0.5,0
1.0,0
1.5,0
2.0,0
2.4,1
2.8,0
3.2,1
3.8,1
4.2,1
4.8,1
5.4,1
6.0,1`;

const TWO_FEATURES = `study_hours,practice_tests,passed
0.5,1,0
1.0,2,0
1.2,4,0
1.8,3,0
2.2,5,0
2.5,7,1
2.8,4,0
2.8,4,1
3.0,8,1
3.4,6,1
3.8,5,1
4.1,8,1
4.5,7,1
5.0,9,1
5.4,6,1`;

const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));

type ModelResult = {
  X: number[][];
  y: number[];
  featureNames: string[];
  targetName: string;
  labels: number[];
  coefficients: number[];
  intercept: number;
  probabilities: number[];
  predictions: number[];
  accuracy: number;
  f1: number;
  logLoss: number;
  mcc: number;
  matrix: number[][];
  means: number[];
  scales: number[];
  mins: number[];
  maxs: number[];
  maxIter: number;
};

export function LogisticRegressionCalculator() {
  const [csv, setCsv] = useState(TWO_FEATURES);
  const deferredCsv = useDeferredValue(csv);
  const [sampleByFeature, setSampleByFeature] = useState<Record<string, number>>({});
  const chartRef = useRef<SVGSVGElement | null>(null);

  const result = useMemo<{ model: ModelResult | null; error: string }>(() => {
    try {
      const parsed = parseNumericTable(deferredCsv);
      if (parsed.headers.length < 2) throw new Error('Provide at least one feature column and one target column.');
      if (parsed.rows.length < 4) throw new Error('Provide at least four observations with both target classes represented.');
      const X = parsed.rows.map((row) => row.slice(0, -1));
      const y = parsed.rows.map((row) => row[row.length - 1]);
      const labels = Array.from(new Set(y)).sort((a, b) => a - b);
      if (labels.length !== 2) throw new Error(`Logistic regression needs exactly two target classes; found ${labels.length}.`);
      const featureNames = parsed.headers.slice(0, -1);
      const targetName = parsed.headers[parsed.headers.length - 1];
      const means = featureNames.map((_, column) => X.reduce((sum, row) => sum + row[column], 0) / X.length);
      const scales = featureNames.map((_, column) => {
        const variance = X.reduce((sum, row) => sum + (row[column] - means[column]) ** 2, 0) / X.length;
        return Math.sqrt(variance) || 1;
      });
      const scaledX = X.map((row) => row.map((value, column) => (value - means[column]) / scales[column]));
      const maxIter = X.length * featureNames.length <= 500 ? 30000 : 5000;
      const classifier = new Linear.LogisticRegression({ learningRate: 0.2, maxIter });
      classifier.fit(scaledX, y);
      const fittedCoefficients = classifier.coef;
      if (!Array.isArray(fittedCoefficients) || Array.isArray(fittedCoefficients[0])) {
        throw new Error('The logistic regression calculator requires a binary fitted model.');
      }
      const scaledCoefficients = fittedCoefficients as number[];
      const zeroDecision = classifier.decisionFunction([new Array(featureNames.length).fill(0)]);
      const scaledIntercept = (zeroDecision as number[])[0];
      const coefficients = scaledCoefficients.map((weight, column) => weight / scales[column]);
      const intercept = scaledIntercept - scaledCoefficients.reduce((sum, weight, column) => sum + (weight * means[column]) / scales[column], 0);
      const probabilities = X.map((row) => sigmoid(intercept + coefficients.reduce((sum, coefficient, column) => sum + coefficient * row[column], 0)));
      const predictions = classifier.predict(scaledX);
      const matrix = Metrics.confusionMatrix(predictions, y, labels);
      const summary = summarizeConfusionMatrix(matrix, labels);
      const mins = featureNames.map((_, column) => Math.min(...X.map((row) => row[column])));
      const maxs = featureNames.map((_, column) => Math.max(...X.map((row) => row[column])));
      return {
        model: {
          X,
          y,
          featureNames,
          targetName,
          labels,
          coefficients,
          intercept,
          probabilities,
          predictions,
          accuracy: Metrics.accuracyScore(predictions, y),
          f1: Metrics.f1Score(predictions, y, { positiveLabel: labels[1] }),
          logLoss: binaryLogLoss(y, probabilities, labels[1]),
          mcc: summary.mcc,
          matrix,
          means,
          scales,
          mins,
          maxs,
          maxIter,
        },
        error: '',
      };
    } catch (caught) {
      return { model: null, error: caught instanceof Error ? caught.message : 'The model could not be fitted.' };
    }
  }, [deferredCsv]);

  const model = result.model;
  const sample = model
    ? model.featureNames.map((name, index) => sampleByFeature[name] ?? model.means[index])
    : [];

  const sampleProbability = model && sample.length === model.featureNames.length
    ? sigmoid(model.intercept + model.coefficients.reduce((sum, coefficient, column) => sum + coefficient * sample[column], 0))
    : null;

  const jsCode = model ? `import { Linear, Metrics } from '@kanaries/ml';

const X = ${JSON.stringify(model.X)};
const y = ${JSON.stringify(model.y)};
const means = ${JSON.stringify(model.means)};
const scales = ${JSON.stringify(model.scales)};
const scaledX = X.map((row) =>
  row.map((value, column) => (value - means[column]) / scales[column]),
);

const model = new Linear.LogisticRegression({
  learningRate: 0.2,
  maxIter: ${model.maxIter},
});
model.fit(scaledX, y);

const predictions = model.predict(scaledX);
const matrix = Metrics.confusionMatrix(predictions, y, ${JSON.stringify(model.labels)});
const accuracy = Metrics.accuracyScore(predictions, y);
const f1 = Metrics.f1Score(predictions, y, { positiveLabel: ${model.labels[1]} });

console.log({ predictions, matrix, accuracy, f1 });` : '';

  const pythonCode = model ? `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X = ${JSON.stringify(model.X)}
y = ${JSON.stringify(model.y)}

model = LogisticRegression(penalty=None, max_iter=${model.maxIter}, tol=1e-12)
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
print(model.coef_, model.intercept_)
print(classification_report(y, predictions))
print(confusion_matrix(y, predictions, labels=${JSON.stringify(model.labels)}))` : '';

  const exportPredictions = () => {
    if (!model) return;
    const header = [...model.featureNames, model.targetName, `p_${model.labels[1]}`, 'prediction'].join(',');
    const rows = model.X.map((row, index) => [...row, model.y[index], model.probabilities[index], model.predictions[index]].join(','));
    downloadText('logistic-regression-predictions.csv', [header, ...rows].join('\n'));
  };

  return (
    <div className={styles.calculator}>
      <div className={styles.grid}>
        <section className={styles.panel}>
          <h2 className={styles.panelTitle}>Training data</h2>
          <DataInput
            value={csv}
            onChange={setCsv}
            label="CSV features and binary target"
            hint="Use one or more feature columns and put the binary target in the final column. Features are standardized during fitting; displayed coefficients are converted back to the original units."
            examples={[
              { label: '2D decision boundary', value: TWO_FEATURES },
              { label: '1D sigmoid curve', value: ONE_FEATURE },
            ]}
          />
          {result.error && <div className={styles.error}>{result.error}</div>}
        </section>

        <section className={styles.panel} aria-live="polite">
          <h2 className={styles.panelTitle}>Fitted model</h2>
          <p className={styles.panelHint}>The greater target label is treated as the positive class. A probability of 0.5 is the decision threshold.</p>
          {model && (
            <>
              <StatCards items={[
                { label: 'Accuracy', value: model.accuracy },
                { label: `F1 (class ${model.labels[1]})`, value: model.f1 },
                { label: 'Log loss', value: model.logLoss },
                { label: 'MCC', value: model.mcc },
              ]} />

              <div className={styles.visualGrid}>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>{model.featureNames.length === 1 ? 'Sigmoid fit' : 'Decision boundary (first two features)'}</h3>
                  <BoundaryChart
                    X={model.X}
                    y={model.y}
                    labels={model.labels}
                    featureNames={model.featureNames}
                    coefficients={model.coefficients}
                    intercept={model.intercept}
                    sample={sample}
                    svgRef={chartRef}
                  />
                </div>
                <div className={styles.chartCard}>
                  <h3 className={styles.chartTitle}>Training confusion matrix</h3>
                  <MatrixHeatmap matrix={model.matrix} labels={model.labels} />
                </div>
              </div>

              <div className={styles.modelTable}>
                <h3 className={styles.chartTitle}>Coefficients and odds ratios</h3>
                <div className={styles.preview}>
                  <table className={styles.table}>
                    <thead><tr><th>Term</th><th>Coefficient</th><th>Odds ratio</th></tr></thead>
                    <tbody>
                      <tr><td>Intercept</td><td>{formatMetric(model.intercept, 5)}</td><td>—</td></tr>
                      {model.featureNames.map((name, index) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{formatMetric(model.coefficients[index], 5)}</td>
                          <td>{formatMetric(Math.exp(model.coefficients[index]), 5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className={styles.modelTable}>
                <h3 className={styles.chartTitle}>Predict a new sample</h3>
                <div className={styles.sliderGrid}>
                  {model.featureNames.map((name, index) => {
                    const range = model.maxs[index] - model.mins[index] || 1;
                    const min = model.mins[index] - range * 0.15;
                    const max = model.maxs[index] + range * 0.15;
                    return (
                      <label className={styles.sliderRow} key={name}>
                        <span>{name}</span>
                        <input
                          type="range"
                          min={min}
                          max={max}
                          step={range / 100}
                          value={sample[index] ?? model.means[index]}
                          onChange={(event) => setSampleByFeature((current) => ({ ...current, [name]: Number(event.target.value) }))}
                        />
                        <span>{formatMetric(sample[index] ?? model.means[index], 2)}</span>
                      </label>
                    );
                  })}
                </div>
                {sampleProbability !== null && (
                  <div className={styles.probability}>
                    P(class {model.labels[1]}) = <strong>{formatMetric(sampleProbability, 4)}</strong>
                    <div className={styles.muted}>Predicted class: {sampleProbability >= 0.5 ? model.labels[1] : model.labels[0]}</div>
                  </div>
                )}
              </div>

              <div className={styles.actions}>
                <button className={styles.button} onClick={exportPredictions} type="button">Download predictions CSV</button>
                <button className={styles.buttonSecondary} onClick={() => downloadSvgAsPng(chartRef.current, 'logistic-regression-fit.png')} type="button">Download chart PNG</button>
              </div>
              <CodeTabs javascript={jsCode} python={pythonCode} />
            </>
          )}
        </section>
      </div>
    </div>
  );
}
