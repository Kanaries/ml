'use client';

import {
  useId,
  useMemo,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import {
  Compose,
  CrossDecomposition,
  Ensemble,
  Kernel,
  Linear,
  MultiOutput,
  Neighbors,
  SVM,
} from '@kanaries/ml';
import styles from './regressionPlayground.module.css';

export type RegressionPlaygroundAlgorithm =
  | 'linear'
  | 'polynomial'
  | 'ridge'
  | 'lasso'
  | 'elasticNet'
  | 'robust'
  | 'bayesian'
  | 'glm'
  | 'randomForest'
  | 'bagging'
  | 'extraTrees'
  | 'adaBoost'
  | 'gradientBoosting'
  | 'xgboost'
  | 'knn'
  | 'radiusNeighbors'
  | 'linearSvr'
  | 'kernelRidge'
  | 'pls'
  | 'transformedTarget'
  | 'regressorChain';

type Dataset = 'wave' | 'curve' | 'outliers' | 'growth';

type Sample = {
  x: number;
  target: number;
  custom?: boolean;
  holdout?: boolean;
};

type RuntimeModel = {
  fit: (features: number[][], targets: number[] | number[][]) => void;
  predict: (features: number[][]) => number[] | number[][];
};

type VariantOption = {
  value: string;
  label: string;
};

type TuningConfig = {
  label: string;
  min: number;
  max: number;
  step: number;
  initial: number;
  format?: (value: number) => string;
};

type FitResult = {
  error: string | null;
  modelLabel: string;
  gridX: number[];
  gridPrediction: number[];
  trainRmse: number;
  validationRmse: number;
  validationR2: number;
};

export type RegressionPlaygroundProps = {
  algorithm: RegressionPlaygroundAlgorithm;
};

const VIEW = {
  width: 700,
  height: 390,
  left: 58,
  right: 22,
  top: 22,
  bottom: 46,
};

const X_DOMAIN: [number, number] = [-3, 3];

const DATASETS: Array<{ value: Dataset; label: string }> = [
  { value: 'wave', label: 'Noisy wave' },
  { value: 'curve', label: 'Curved trend' },
  { value: 'outliers', label: 'Linear + outliers' },
  { value: 'growth', label: 'Positive growth' },
];

const ALGORITHM_LABELS: Record<RegressionPlaygroundAlgorithm, string> = {
  linear: 'LinearRegression',
  polynomial: 'PolynomialRegression',
  ridge: 'RidgeRegression',
  lasso: 'LassoRegression',
  elasticNet: 'ElasticNet',
  robust: 'Robust regression',
  bayesian: 'Bayesian regression',
  glm: 'Generalized linear model',
  randomForest: 'RandomForestRegressor',
  bagging: 'BaggingRegressor',
  extraTrees: 'ExtraTreesRegressor',
  adaBoost: 'AdaBoostRegressor',
  gradientBoosting: 'GradientBoostingRegressor',
  xgboost: 'XGBoostRegressor',
  knn: 'KNeighborsRegressor',
  radiusNeighbors: 'RadiusNeighborsRegressor',
  linearSvr: 'LinearSVR',
  kernelRidge: 'KernelRidge',
  pls: 'PLSRegression',
  transformedTarget: 'TransformedTargetRegressor',
  regressorChain: 'RegressorChain',
};

const VARIANTS: Partial<Record<RegressionPlaygroundAlgorithm, VariantOption[]>> = {
  robust: [
    { value: 'huber', label: 'Huber' },
    { value: 'ransac', label: 'RANSAC' },
    { value: 'theilSen', label: 'Theil–Sen' },
    { value: 'quantile', label: 'Quantile' },
  ],
  bayesian: [
    { value: 'bayesianRidge', label: 'Bayesian Ridge' },
    { value: 'ard', label: 'ARD Regression' },
  ],
  glm: [
    { value: 'poisson', label: 'Poisson' },
    { value: 'gamma', label: 'Gamma' },
    { value: 'tweedie', label: 'Tweedie' },
  ],
  transformedTarget: [
    { value: 'log1p', label: 'log1p → expm1' },
    { value: 'identity', label: 'Identity transform' },
  ],
};

function createRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random: () => number) {
  const u = Math.max(random(), Number.EPSILON);
  const v = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function signal(dataset: Dataset, x: number) {
  if (dataset === 'curve') return 0.38 * x * x - 0.48 * x - 0.75;
  if (dataset === 'outliers') return 0.78 * x + 0.25;
  if (dataset === 'growth') return 0.42 * Math.exp((x + 3) * 0.42) + 0.15;
  return 0.92 * Math.sin(x * 1.55) + 0.2 * x;
}

function generateSamples(
  dataset: Dataset,
  count: number,
  noise: number,
  seed: number,
  positiveTargets: boolean,
): Sample[] {
  const random = createRandom(seed);
  return Array.from({ length: count }, (_, index) => {
    const x = X_DOMAIN[0] + random() * (X_DOMAIN[1] - X_DOMAIN[0]);
    const contamination = dataset === 'outliers' && index % 9 === 0
      ? (index % 18 === 0 ? 2.25 : -2.25)
      : 0;
    let target = signal(dataset, x) + gaussian(random) * noise + contamination;
    if (positiveTargets) target = Math.max(0.08, target + (dataset === 'growth' ? 0 : 3.15));
    return { x, target, holdout: index % 5 === 0 };
  }).sort((a, b) => a.x - b.x);
}

function tuningConfig(algorithm: RegressionPlaygroundAlgorithm, variant: string): TuningConfig | null {
  if (algorithm === 'polynomial') return { label: 'Polynomial degree', min: 1, max: 7, step: 1, initial: 3 };
  if (['ridge', 'lasso', 'elasticNet', 'glm', 'regressorChain'].includes(algorithm)) {
    return { label: 'Regularization α', min: 0, max: 2, step: 0.05, initial: 0.2, format: value => value.toFixed(2) };
  }
  if (['randomForest', 'bagging', 'extraTrees', 'adaBoost', 'gradientBoosting', 'xgboost'].includes(algorithm)) {
    return { label: 'Number of estimators', min: 5, max: 60, step: 5, initial: 25 };
  }
  if (algorithm === 'knn') return { label: 'Neighbors (k)', min: 1, max: 18, step: 1, initial: 5 };
  if (algorithm === 'radiusNeighbors') return { label: 'Neighbor radius', min: 0.2, max: 1.8, step: 0.05, initial: 0.7, format: value => value.toFixed(2) };
  if (algorithm === 'linearSvr') return { label: 'Penalty C', min: 0.1, max: 4, step: 0.1, initial: 1, format: value => value.toFixed(1) };
  if (algorithm === 'kernelRidge') return { label: 'RBF gamma', min: 0.1, max: 3, step: 0.1, initial: 0.8, format: value => value.toFixed(1) };
  if (algorithm === 'pls') return { label: 'PLS components', min: 1, max: 2, step: 1, initial: 2 };
  if (algorithm === 'bayesian') return { label: 'Maximum iterations', min: 25, max: 300, step: 25, initial: 150 };
  if (algorithm === 'robust' && variant === 'huber') return { label: 'Huber epsilon', min: 1.05, max: 2.5, step: 0.05, initial: 1.35, format: value => value.toFixed(2) };
  if (algorithm === 'robust' && variant === 'ransac') return { label: 'Residual threshold', min: 0.1, max: 1.5, step: 0.05, initial: 0.45, format: value => value.toFixed(2) };
  if (algorithm === 'robust' && variant === 'theilSen') return { label: 'Subpopulation cap', min: 40, max: 400, step: 40, initial: 160 };
  if (algorithm === 'robust' && variant === 'quantile') return { label: 'Target quantile', min: 0.1, max: 0.9, step: 0.05, initial: 0.5, format: value => value.toFixed(2) };
  return null;
}

function modelLabel(algorithm: RegressionPlaygroundAlgorithm, variant: string) {
  const selected = VARIANTS[algorithm]?.find(option => option.value === variant);
  return selected?.label ?? ALGORITHM_LABELS[algorithm];
}

function createModel(
  algorithm: RegressionPlaygroundAlgorithm,
  variant: string,
  tuning: number,
  seed: number,
): RuntimeModel {
  let model: unknown;

  switch (algorithm) {
    case 'linear':
      model = new Linear.LinearRegression();
      break;
    case 'polynomial':
      model = new Linear.PolynomialRegression({ degree: Math.round(tuning) });
      break;
    case 'ridge':
      model = new Linear.RidgeRegression({ alpha: tuning });
      break;
    case 'lasso':
      model = new Linear.LassoRegression({ alpha: tuning, maxIter: 800, tol: 1e-6 });
      break;
    case 'elasticNet':
      model = new Linear.ElasticNet({ alpha: tuning, l1Ratio: 0.5, maxIter: 800, tol: 1e-6 });
      break;
    case 'robust':
      if (variant === 'ransac') model = new Linear.RANSACRegressor({ residualThreshold: tuning, maxTrials: 60, randomState: seed });
      else if (variant === 'theilSen') model = new Linear.TheilSenRegressor({ maxSubpopulation: Math.round(tuning), randomState: seed });
      else if (variant === 'quantile') model = new Linear.QuantileRegressor({ quantile: tuning, alpha: 0.02, maxIter: 900, tol: 1e-5 });
      else model = new Linear.HuberRegressor({ epsilon: tuning, maxIter: 180 });
      break;
    case 'bayesian':
      model = variant === 'ard'
        ? new Linear.ARDRegression({ maxIter: Math.round(tuning) })
        : new Linear.BayesianRidge({ maxIter: Math.round(tuning) });
      break;
    case 'glm':
      if (variant === 'gamma') model = new Linear.GammaRegressor({ alpha: tuning, maxIter: 120 });
      else if (variant === 'tweedie') model = new Linear.TweedieRegressor({ power: 1.5, alpha: tuning, link: 'log', maxIter: 120 });
      else model = new Linear.PoissonRegressor({ alpha: tuning, maxIter: 120 });
      break;
    case 'randomForest':
      model = new Ensemble.RandomForestRegressor({ nEstimators: Math.round(tuning), maxDepth: 5, randomState: seed });
      break;
    case 'bagging':
      model = new Ensemble.BaggingRegressor({ nEstimators: Math.round(tuning), maxSamples: 0.82, randomState: seed });
      break;
    case 'extraTrees':
      model = new Ensemble.ExtraTreesRegressor({ nEstimators: Math.round(tuning), max_depth: 5, randomState: seed });
      break;
    case 'adaBoost':
      model = new Ensemble.AdaBoostRegressor({ nEstimators: Math.round(tuning), learningRate: 0.7, randomState: seed });
      break;
    case 'gradientBoosting':
      model = new Ensemble.GradientBoostingRegressor({ nEstimators: Math.round(tuning), learningRate: 0.08, maxDepth: 3, randomState: seed });
      break;
    case 'xgboost':
      model = new Ensemble.XGBoostRegressor({ nEstimators: Math.round(tuning), learningRate: 0.18, maxDepth: 3, randomState: seed });
      break;
    case 'knn':
      model = new Neighbors.KNeighborsRegressor({ nNeighbors: Math.round(tuning), weights: 'distance' });
      break;
    case 'radiusNeighbors':
      model = new Neighbors.RadiusNeighborsRegressor({ radius: tuning, weights: 'distance' });
      break;
    case 'linearSvr':
      model = new SVM.LinearSVR({ C: tuning, epsilon: 0.08, maxIter: 500, randomState: seed });
      break;
    case 'kernelRidge':
      model = new Kernel.KernelRidge({ kernel: 'rbf', gamma: tuning, alpha: 0.18 });
      break;
    case 'pls':
      model = new CrossDecomposition.PLSRegression({ nComponents: Math.round(tuning) });
      break;
    case 'transformedTarget':
      model = variant === 'identity'
        ? new Compose.TransformedTargetRegressor({ regressor: new Linear.RidgeRegression({ alpha: 0.08 }) })
        : new Compose.TransformedTargetRegressor({
            regressor: new Linear.RidgeRegression({ alpha: 0.08 }),
            func: 'log1p',
            inverseFunc: 'expm1',
          });
      break;
    case 'regressorChain':
      model = new MultiOutput.RegressorChain({
        estimator: new Linear.RidgeRegression({ alpha: tuning }),
        order: [0, 1],
      });
      break;
  }

  return model as RuntimeModel;
}

function featureRows(algorithm: RegressionPlaygroundAlgorithm, values: number[]) {
  if (algorithm === 'pls') return values.map(x => [x, x * x / 3]);
  return values.map(x => [x]);
}

function targetsFor(algorithm: RegressionPlaygroundAlgorithm, samples: Sample[]) {
  if (algorithm === 'pls' || algorithm === 'regressorChain') {
    return samples.map(sample => [
      sample.target,
      0.62 * sample.target + 0.34 * sample.x + 0.12 * Math.sin(sample.x),
    ]);
  }
  return samples.map(sample => sample.target);
}

function firstOutput(prediction: number[] | number[][]) {
  if (prediction.length === 0) return [];
  return Array.isArray(prediction[0])
    ? (prediction as number[][]).map(row => row[0])
    : prediction as number[];
}

function rmse(expected: number[], predicted: number[]) {
  const pairs = expected
    .map((value, index) => [value, predicted[index]] as const)
    .filter(([, value]) => Number.isFinite(value));
  if (pairs.length === 0) return Number.NaN;
  return Math.sqrt(pairs.reduce((sum, [actual, prediction]) => sum + (actual - prediction) ** 2, 0) / pairs.length);
}

function r2(expected: number[], predicted: number[]) {
  const pairs = expected
    .map((value, index) => [value, predicted[index]] as const)
    .filter(([, value]) => Number.isFinite(value));
  if (pairs.length < 2) return Number.NaN;
  const mean = pairs.reduce((sum, [value]) => sum + value, 0) / pairs.length;
  const total = pairs.reduce((sum, [value]) => sum + (value - mean) ** 2, 0);
  const residual = pairs.reduce((sum, [value, prediction]) => sum + (value - prediction) ** 2, 0);
  return total === 0 ? Number.NaN : 1 - residual / total;
}

function fitModel(
  algorithm: RegressionPlaygroundAlgorithm,
  variant: string,
  tuning: number,
  samples: Sample[],
  seed: number,
): FitResult {
  const modelName = modelLabel(algorithm, variant);
  const train = samples.filter(sample => sample.custom || !sample.holdout);
  const validation = samples.filter(sample => !sample.custom && sample.holdout);
  const gridX = Array.from({ length: 181 }, (_, index) => X_DOMAIN[0] + index / 180 * (X_DOMAIN[1] - X_DOMAIN[0]));

  try {
    const model = createModel(algorithm, variant, tuning, seed);
    model.fit(featureRows(algorithm, train.map(sample => sample.x)), targetsFor(algorithm, train));
    const trainPrediction = firstOutput(model.predict(featureRows(algorithm, train.map(sample => sample.x))));
    const validationPrediction = firstOutput(model.predict(featureRows(algorithm, validation.map(sample => sample.x))));
    const gridPrediction = firstOutput(model.predict(featureRows(algorithm, gridX)));
    return {
      error: null,
      modelLabel: modelName,
      gridX,
      gridPrediction,
      trainRmse: rmse(train.map(sample => sample.target), trainPrediction),
      validationRmse: rmse(validation.map(sample => sample.target), validationPrediction),
      validationR2: r2(validation.map(sample => sample.target), validationPrediction),
    };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : 'The model could not be fitted.',
      modelLabel: modelName,
      gridX,
      gridPrediction: [],
      trainRmse: Number.NaN,
      validationRmse: Number.NaN,
      validationR2: Number.NaN,
    };
  }
}

function yDomain(samples: Sample[], prediction: number[]): [number, number] {
  const values = [...samples.map(sample => sample.target), ...prediction].filter(Number.isFinite);
  if (values.length === 0) return [-1, 1];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const padding = Math.max((maximum - minimum) * 0.16, 0.35);
  return [minimum - padding, maximum + padding];
}

function xScale(value: number) {
  const width = VIEW.width - VIEW.left - VIEW.right;
  return Number((VIEW.left + (value - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0]) * width).toFixed(4));
}

function yScale(value: number, domain: [number, number]) {
  const height = VIEW.height - VIEW.top - VIEW.bottom;
  return Number((VIEW.top + (domain[1] - value) / (domain[1] - domain[0]) * height).toFixed(4));
}

function xFromView(value: number) {
  const width = VIEW.width - VIEW.left - VIEW.right;
  return X_DOMAIN[0] + (value - VIEW.left) / width * (X_DOMAIN[1] - X_DOMAIN[0]);
}

function yFromView(value: number, domain: [number, number]) {
  const height = VIEW.height - VIEW.top - VIEW.bottom;
  return domain[1] - (value - VIEW.top) / height * (domain[1] - domain[0]);
}

function predictionPaths(gridX: number[], prediction: number[], domain: [number, number]) {
  const paths: string[] = [];
  let current = '';
  prediction.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      if (current) paths.push(current);
      current = '';
      return;
    }
    const command = current ? 'L' : 'M';
    current += `${command}${xScale(gridX[index]).toFixed(2)},${yScale(value, domain).toFixed(2)} `;
  });
  if (current) paths.push(current);
  return paths;
}

function formatMetric(value: number, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

export function RegressionPlayground({ algorithm }: RegressionPlaygroundProps) {
  const variants = VARIANTS[algorithm] ?? [];
  const [dataset, setDataset] = useState<Dataset>(algorithm === 'glm' || algorithm === 'transformedTarget' ? 'growth' : 'wave');
  const [noise, setNoise] = useState(0.18);
  const [sampleCount, setSampleCount] = useState(48);
  const [seed, setSeed] = useState(17);
  const [variant, setVariant] = useState(variants[0]?.value ?? algorithm);
  const initialTuning = tuningConfig(algorithm, variants[0]?.value ?? algorithm)?.initial ?? 1;
  const [tuning, setTuning] = useState(initialTuning);
  const [customSamples, setCustomSamples] = useState<Sample[]>([]);
  const [coordinateX, setCoordinateX] = useState('0');
  const [coordinateY, setCoordinateY] = useState('0');
  const [coordinateError, setCoordinateError] = useState<string | null>(null);
  const positiveTargets = algorithm === 'glm' || algorithm === 'transformedTarget';
  const clipId = `${useId().replaceAll(':', '')}-regression-clip`;
  const coordinateErrorId = `${clipId}-coordinate-error`;
  const tune = tuningConfig(algorithm, variant);

  const generatedSamples = useMemo(
    () => generateSamples(dataset, sampleCount, noise, seed, positiveTargets),
    [dataset, noise, positiveTargets, sampleCount, seed],
  );
  const samples = useMemo(
    () => [...generatedSamples, ...customSamples].sort((a, b) => a.x - b.x),
    [customSamples, generatedSamples],
  );
  const fitted = useMemo(
    () => fitModel(algorithm, variant, tuning, samples, seed),
    [algorithm, samples, seed, tuning, variant],
  );
  const domain = useMemo(() => yDomain(samples, fitted.gridPrediction), [fitted.gridPrediction, samples]);
  const paths = useMemo(
    () => predictionPaths(fitted.gridX, fitted.gridPrediction, domain),
    [domain, fitted.gridPrediction, fitted.gridX],
  );
  const xTicks = [-3, -2, -1, 0, 1, 2, 3];
  const yTicks = Array.from({ length: 5 }, (_, index) => domain[0] + index / 4 * (domain[1] - domain[0]));
  const plotBottom = VIEW.height - VIEW.bottom;
  const plotRight = VIEW.width - VIEW.right;

  const addObservation = (x: number, target: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(target)) {
      setCoordinateError('Enter finite x and y values.');
      return;
    }
    if (x < X_DOMAIN[0] || x > X_DOMAIN[1]) {
      setCoordinateError(`x must be between ${X_DOMAIN[0]} and ${X_DOMAIN[1]}.`);
      return;
    }
    if (positiveTargets && target <= 0) {
      setCoordinateError('y must be greater than 0 for this model.');
      return;
    }
    setCoordinateError(null);
    setCustomSamples(current => [...current, { x, target, custom: true }]);
  };

  const handlePlotClick = (event: ReactMouseEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = (event.clientX - bounds.left) / bounds.width * VIEW.width;
    const viewY = (event.clientY - bounds.top) / bounds.height * VIEW.height;
    if (viewX < VIEW.left || viewX > plotRight || viewY < VIEW.top || viewY > plotBottom) return;
    addObservation(xFromView(viewX), yFromView(viewY, domain));
  };

  const resetGeneratedData = (nextSeed = seed) => {
    setSeed(nextSeed);
    setCustomSamples([]);
    setCoordinateError(null);
  };

  return (
    <section className={styles.playground} aria-label={`${ALGORITHM_LABELS[algorithm]} interactive regression playground`}>
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Live browser model</span>
          <h3>{fitted.modelLabel} playground</h3>
          <p>Adjust the data and model, then click the chart to add a training observation.</p>
        </div>
        <div className={fitted.error ? styles.statusError : styles.status}>
          <span /> {fitted.error ? 'Fit failed' : 'Fitted with @kanaries/ml'}
        </div>
      </div>

      <div className={styles.layout}>
        <div className={styles.chartCard}>
          <svg
            className={styles.chart}
            viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
            role="img"
            aria-label={`${fitted.modelLabel} fitted prediction and regression observations`}
            onClick={handlePlotClick}
          >
            <defs><clipPath id={clipId}><rect x={VIEW.left} y={VIEW.top} width={plotRight - VIEW.left} height={plotBottom - VIEW.top} /></clipPath></defs>
            <rect x={VIEW.left} y={VIEW.top} width={plotRight - VIEW.left} height={plotBottom - VIEW.top} className={styles.plotBackground} />
            {xTicks.map(tick => (
              <g key={`x-${tick}`}>
                <line x1={xScale(tick)} x2={xScale(tick)} y1={VIEW.top} y2={plotBottom} className={styles.gridLine} />
                <text x={xScale(tick)} y={plotBottom + 20} textAnchor="middle" className={styles.axisText}>{tick}</text>
              </g>
            ))}
            {yTicks.map((tick, index) => (
              <g key={`y-${index}`}>
                <line x1={VIEW.left} x2={plotRight} y1={yScale(tick, domain)} y2={yScale(tick, domain)} className={styles.gridLine} />
                <text x={VIEW.left - 10} y={yScale(tick, domain) + 4} textAnchor="end" className={styles.axisText}>{tick.toFixed(1)}</text>
              </g>
            ))}
            <line x1={VIEW.left} x2={plotRight} y1={plotBottom} y2={plotBottom} className={styles.axisLine} />
            <line x1={VIEW.left} x2={VIEW.left} y1={VIEW.top} y2={plotBottom} className={styles.axisLine} />
            <text x={(VIEW.left + plotRight) / 2} y={VIEW.height - 8} textAnchor="middle" className={styles.axisLabel}>feature x</text>
            <text x="14" y={(VIEW.top + plotBottom) / 2} textAnchor="middle" className={styles.axisLabel} transform={`rotate(-90 14 ${(VIEW.top + plotBottom) / 2})`}>target y</text>
            <g clipPath={`url(#${clipId})`}>
              {paths.map((path, index) => <path key={index} d={path} className={styles.predictionLine} />)}
              {samples.map((sample, index) => (
                <circle
                  key={`${sample.x}-${sample.target}-${index}`}
                  cx={xScale(sample.x)}
                  cy={yScale(sample.target, domain)}
                  r={sample.custom ? 5.2 : 3.8}
                  className={sample.custom ? styles.customPoint : sample.holdout ? styles.validationPoint : styles.trainingPoint}
                />
              ))}
            </g>
          </svg>
          <div className={styles.legend}>
            <span><i className={styles.lineKey} /> prediction</span>
            <span><i className={styles.trainKey} /> training</span>
            <span><i className={styles.validationKey} /> holdout</span>
            <span><i className={styles.customKey} /> your points</span>
          </div>
          {fitted.error && <div className={styles.error} role="alert">Fit error: {fitted.error}</div>}
        </div>

        <aside className={styles.controls}>
          <label>
            <span>Dataset</span>
            <select value={dataset} onChange={event => { setDataset(event.target.value as Dataset); setCustomSamples([]); }}>
              {DATASETS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </label>

          {variants.length > 0 && (
            <label>
              <span>Estimator</span>
              <select value={variant} onChange={event => {
                const next = event.target.value;
                setVariant(next);
                setTuning(tuningConfig(algorithm, next)?.initial ?? 1);
              }}>
                {variants.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
            </label>
          )}

          {tune && (
            <label>
              <span>{tune.label} <strong>{tune.format?.(tuning) ?? tuning}</strong></span>
              <input type="range" min={tune.min} max={tune.max} step={tune.step} value={tuning} onChange={event => setTuning(Number(event.target.value))} />
            </label>
          )}

          <label>
            <span>Noise <strong>{noise.toFixed(2)}</strong></span>
            <input type="range" min="0" max="0.65" step="0.01" value={noise} onChange={event => { setNoise(Number(event.target.value)); setCustomSamples([]); }} />
          </label>

          <label>
            <span>Samples <strong>{sampleCount}</strong></span>
            <input type="range" min="24" max="80" step="4" value={sampleCount} onChange={event => { setSampleCount(Number(event.target.value)); setCustomSamples([]); }} />
          </label>

          <div className={styles.actions}>
            <button type="button" onClick={() => resetGeneratedData(seed + 1)}>New sample</button>
            <button type="button" className={styles.secondaryButton} onClick={() => setCustomSamples([])} disabled={customSamples.length === 0}>Clear points</button>
          </div>

          <fieldset
            className={styles.coordinateInput}
            aria-label="Add an observation by coordinates"
            aria-describedby={coordinateError ? coordinateErrorId : undefined}
          >
            <legend>Add exact point</legend>
            <label><span>x</span><input type="number" min={X_DOMAIN[0]} max={X_DOMAIN[1]} step="0.1" value={coordinateX} onChange={event => { setCoordinateX(event.target.value); setCoordinateError(null); }} /></label>
            <label><span>y</span><input type="number" step="0.1" value={coordinateY} onChange={event => { setCoordinateY(event.target.value); setCoordinateError(null); }} /></label>
            <button type="button" onClick={() => {
              if (coordinateX.trim() === '' || coordinateY.trim() === '') {
                setCoordinateError('Enter both x and y values.');
                return;
              }
              addObservation(Number(coordinateX), Number(coordinateY));
            }}>Add</button>
            {coordinateError && <p id={coordinateErrorId} className={styles.coordinateError} role="alert">{coordinateError}</p>}
          </fieldset>
        </aside>
      </div>

      <div className={styles.metrics} aria-live="polite">
        <div><span>Train RMSE</span><strong>{formatMetric(fitted.trainRmse)}</strong></div>
        <div><span>Holdout RMSE</span><strong>{formatMetric(fitted.validationRmse)}</strong></div>
        <div><span>Holdout R²</span><strong>{formatMetric(fitted.validationR2)}</strong></div>
        <div><span>Custom points</span><strong>{customSamples.length}</strong></div>
      </div>
    </section>
  );
}
