'use client';

import {
  useEffect,
  useId,
  useMemo,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import { Clusters } from '@kanaries/ml';
import { makeBlobs, makeMoons, type Point2D } from './playground/data';
import styles from './clusteringPlayground.module.css';

export type ClusteringPlaygroundAlgorithm =
  | 'dbscan'
  | 'hdbscan'
  | 'meanShift'
  | 'optics'
  | 'kmeansPlusPlus'
  | 'advanced';

type AdvancedAlgorithm = 'birch' | 'affinityPropagation' | 'bisectingKMeans';
type Dataset = 'blobs' | 'moons' | 'variableDensity';
type ActiveAlgorithm = Exclude<ClusteringPlaygroundAlgorithm, 'advanced'> | AdvancedAlgorithm;

type FittedClustering = {
  labels: number[];
  centers: number[][];
  centerLabels?: number[];
  strengths?: number[];
  seedIndices?: number[];
  detailLabel: string;
  detailValue: string;
};

export type ClusteringPlaygroundProps = {
  algorithm: ClusteringPlaygroundAlgorithm;
};

const VIEW = {
  width: 640,
  height: 390,
  left: 50,
  right: 18,
  top: 18,
  bottom: 42,
};

const X_DOMAIN: [number, number] = [-3.2, 3.2];
const Y_DOMAIN: [number, number] = [-3, 3];
const COLORS = ['#236a9d', '#c65f46', '#378665', '#9a6ab1', '#d1902d', '#327f89', '#b65a78', '#6d7f35'];
const NOISE_COLOR = '#8b8f95';

function clusterColor(label: number) {
  if (label < COLORS.length) return COLORS[label];
  const hue = (label * 137.508 + 202) % 360;
  const saturation = 52 + (label % 3) * 6;
  const lightness = 42 + (label % 2) * 9;
  return `hsl(${hue.toFixed(1)} ${saturation}% ${lightness}%)`;
}

const ALGORITHM_NAMES: Record<ActiveAlgorithm, string> = {
  dbscan: 'DBSCAN',
  hdbscan: 'HDBSCAN',
  meanShift: 'Mean Shift',
  optics: 'OPTICS',
  kmeansPlusPlus: 'k-means++',
  birch: 'Birch',
  affinityPropagation: 'Affinity Propagation',
  bisectingKMeans: 'Bisecting K-Means',
};

const ALGORITHM_BADGES: Record<ActiveAlgorithm, string> = {
  dbscan: 'Clusters.DBScan',
  hdbscan: 'Clusters.HDBScan',
  meanShift: 'Clusters.MeanShift',
  optics: 'Clusters.OPTICS',
  kmeansPlusPlus: 'Clusters.kmeansPlusPlus',
  birch: 'Clusters.Birch',
  affinityPropagation: 'Clusters.AffinityPropagation',
  bisectingKMeans: 'Clusters.BisectingKMeans',
};

const ALGORITHM_DESCRIPTIONS: Record<ActiveAlgorithm, string> = {
  dbscan: 'Change the neighborhood definition and watch dense regions connect while sparse observations become noise.',
  hdbscan: 'Change the minimum stable group and compare membership strength across clusters with uneven density.',
  meanShift: 'Move the bandwidth to merge or separate density peaks without choosing a cluster count in advance.',
  optics: 'Tune the extraction radius and see how the density ordering becomes flat clusters and noise.',
  kmeansPlusPlus: 'Reroll the weighted seeding process and see how far-away observations are favored as starting centroids.',
  birch: 'Adjust the compression threshold and final cluster count while a Birch CF tree summarizes the observations.',
  affinityPropagation: 'Adjust exemplar preference and damping while observations exchange messages to choose representatives.',
  bisectingKMeans: 'Choose how many leaves to create and which partition should be split at each step.',
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

function createVariableDensityData(noise: number, seed: number): Point2D[] {
  const random = createRandom(seed);
  const dense = Array.from({ length: 38 }, () => ({
    x: -1.35 + gaussian(random) * (0.13 + noise * 0.5),
    y: -0.55 + gaussian(random) * (0.13 + noise * 0.5),
  }));
  const sparse = Array.from({ length: 42 }, () => ({
    x: 1.15 + gaussian(random) * (0.38 + noise),
    y: 0.75 + gaussian(random) * (0.34 + noise),
  }));
  const outliers = Array.from({ length: 8 }, () => ({
    x: random() * 5.6 - 2.8,
    y: random() * 5 - 2.5,
  }));
  return [...dense, ...sparse, ...outliers];
}

function createDataset(dataset: Dataset, noise: number, seed: number): Point2D[] {
  if (dataset === 'moons') {
    return makeMoons(88, 0.035 + noise * 0.7, seed).map((point) => ({
      x: point.x * 1.55 - 0.8,
      y: point.y * 1.6,
    }));
  }
  if (dataset === 'variableDensity') return createVariableDensityData(noise, seed);
  return makeBlobs(
    84,
    [[-1.65, -1.05], [1.55, -0.65], [0.15, 1.55]],
    0.22 + noise,
    seed,
  ).map(({ x, y }) => ({ x, y }));
}

function initialDataset(algorithm: ClusteringPlaygroundAlgorithm): Dataset {
  if (algorithm === 'hdbscan') return 'variableDensity';
  if (algorithm === 'dbscan' || algorithm === 'optics') return 'moons';
  return 'blobs';
}

function nearestCenterLabels(features: number[][], centers: number[][]) {
  return features.map((row) => centers.reduce(
    (best, center, index) => {
      const distance = row.reduce((sum, value, feature) => sum + (value - center[feature]) ** 2, 0);
      return distance < best.distance ? { index, distance } : best;
    },
    { index: 0, distance: Number.POSITIVE_INFINITY },
  ).index);
}

function fitClustering(
  algorithm: ActiveAlgorithm,
  features: number[][],
  parameters: {
    eps: number;
    minSamples: number;
    minClusterSize: number;
    bandwidth: number;
    k: number;
    threshold: number;
    damping: number;
    preference: number;
    strategy: 'biggestInertia' | 'largestCluster';
    modelSeed: number;
  },
): FittedClustering {
  if (algorithm === 'dbscan') {
    const model = new Clusters.DBScan({ eps: parameters.eps, minSamples: parameters.minSamples });
    return {
      labels: model.fitPredict(features),
      centers: [],
      detailLabel: 'Radius (eps)',
      detailValue: parameters.eps.toFixed(2),
    };
  }

  if (algorithm === 'hdbscan') {
    const model = new Clusters.HDBScan({
      min_cluster_size: parameters.minClusterSize,
      min_samples: parameters.minSamples,
    });
    const labels = model.fitPredict(features);
    const strengths = model.getProbabilities();
    const assigned = strengths.filter((_, index) => labels[index] !== -1);
    const meanStrength = assigned.length === 0
      ? 0
      : assigned.reduce((sum, value) => sum + value, 0) / assigned.length;
    return {
      labels,
      centers: [],
      strengths,
      detailLabel: 'Mean strength',
      detailValue: meanStrength.toFixed(2),
    };
  }

  if (algorithm === 'meanShift') {
    const model = new Clusters.MeanShift({ bandwidth: parameters.bandwidth, max_iter: 80 });
    return {
      labels: model.fitPredict(features),
      centers: model.getCentroids(),
      detailLabel: 'Bandwidth',
      detailValue: parameters.bandwidth.toFixed(2),
    };
  }

  if (algorithm === 'optics') {
    const model = new Clusters.OPTICS({
      min_samples: parameters.minSamples,
      max_eps: Math.max(parameters.eps, 1.2),
      eps: parameters.eps,
    });
    return {
      labels: model.fitPredict(features),
      centers: [],
      detailLabel: 'Extract eps',
      detailValue: parameters.eps.toFixed(2),
    };
  }

  if (algorithm === 'kmeansPlusPlus') {
    const seeded = Clusters.kmeansPlusPlus(
      features,
      parameters.k,
      undefined,
      createRandom(parameters.modelSeed),
    );
    return {
      labels: nearestCenterLabels(features, seeded.centers),
      centers: seeded.centers,
      seedIndices: seeded.indices,
      detailLabel: 'Selected seeds',
      detailValue: seeded.indices.map((index) => index + 1).join(', '),
    };
  }

  if (algorithm === 'birch') {
    const model = new Clusters.Birch({
      threshold: parameters.threshold,
      branchingFactor: 16,
      nClusters: parameters.k,
    });
    return {
      labels: model.fitPredict(features),
      centers: model.subclusterCenters,
      centerLabels: model.subclusterLabels,
      detailLabel: 'CF subclusters',
      detailValue: String(model.subclusterCenters.length),
    };
  }

  if (algorithm === 'affinityPropagation') {
    const model = new Clusters.AffinityPropagation({
      damping: parameters.damping,
      preference: parameters.preference,
      maxIter: 160,
      convergenceIter: 12,
      randomState: parameters.modelSeed,
    });
    const labels = model.fitPredict(features);
    return {
      labels,
      centers: model.clusterCenters,
      detailLabel: 'Iterations',
      detailValue: String(model.nIter),
    };
  }

  const model = new Clusters.BisectingKMeans({
    nClusters: parameters.k,
    bisectingStrategy: parameters.strategy,
    randomState: parameters.modelSeed,
    nInit: 4,
  });
  return {
    labels: model.fitPredict(features),
    centers: model.clusterCenters,
    detailLabel: 'Inertia',
    detailValue: model.inertia.toFixed(2),
  };
}

function xScale(value: number) {
  const plotWidth = VIEW.width - VIEW.left - VIEW.right;
  return VIEW.left + ((value - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * plotWidth;
}

function yScale(value: number) {
  const plotHeight = VIEW.height - VIEW.top - VIEW.bottom;
  return VIEW.top + ((Y_DOMAIN[1] - value) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * plotHeight;
}

function algorithmInsight(algorithm: ActiveAlgorithm, clusterCount: number, noiseCount: number, dataset: Dataset) {
  if (clusterCount === 0) {
    return 'No stable cluster is selected with these settings. Relax the density constraint or regenerate a denser sample.';
  }
  if (algorithm === 'kmeansPlusPlus') {
    return 'The numbered observations are initial seeds, not final centroids. Reroll to see how squared distance changes each weighted choice.';
  }
  if (algorithm === 'hdbscan') {
    return 'Point opacity shows HDBSCAN membership strength. Pale observations sit closer to the edge of their selected cluster.';
  }
  if (algorithm === 'birch') {
    return 'Outlined diamonds are CF subcluster centers. The final colors come from clustering those compressed summaries.';
  }
  if ((algorithm === 'dbscan' || algorithm === 'optics') && noiseCount > 0) {
    return `${noiseCount} gray observation${noiseCount === 1 ? ' is' : 's are'} currently labeled as noise. Increase the radius to connect more neighborhoods.`;
  }
  if (algorithm === 'meanShift' && dataset === 'moons') {
    return 'Mean Shift follows density modes rather than curved manifolds, so a moons dataset may break into several local peaks.';
  }
  if (algorithm === 'affinityPropagation') {
    return 'Outlined diamonds are the exemplars selected by message passing. A higher preference usually allows more exemplars.';
  }
  if (algorithm === 'bisectingKMeans') {
    return 'Each additional color is created by splitting one existing partition with a two-center K-Means fit.';
  }
  return 'Click inside the chart to add an observation. The model refits immediately, so you can test how a boundary point changes the result.';
}

export function ClusteringPlayground({ algorithm }: ClusteringPlaygroundProps) {
  const [mounted, setMounted] = useState(false);
  const [dataset, setDataset] = useState<Dataset>(() => initialDataset(algorithm));
  const [noise, setNoise] = useState(0.13);
  const [dataSeed, setDataSeed] = useState(19);
  const [modelSeed, setModelSeed] = useState(43);
  const [customPoints, setCustomPoints] = useState<Point2D[]>([]);
  const [advancedAlgorithm, setAdvancedAlgorithm] = useState<AdvancedAlgorithm>('birch');
  const [eps, setEps] = useState(0.42);
  const [minSamples, setMinSamples] = useState(5);
  const [minClusterSize, setMinClusterSize] = useState(7);
  const [bandwidth, setBandwidth] = useState(0.72);
  const [k, setK] = useState(3);
  const [threshold, setThreshold] = useState(0.38);
  const [damping, setDamping] = useState(0.68);
  const [preference, setPreference] = useState(-3.2);
  const [strategy, setStrategy] = useState<'biggestInertia' | 'largestCluster'>('biggestInertia');
  const [manualX, setManualX] = useState(0);
  const [manualY, setManualY] = useState(0);
  const rawId = useId();
  const controlId = rawId.replace(/[^a-zA-Z0-9_-]/g, '');

  useEffect(() => setMounted(true), []);

  const activeAlgorithm: ActiveAlgorithm = algorithm === 'advanced' ? advancedAlgorithm : algorithm;
  const basePoints = useMemo(
    () => createDataset(dataset, noise, dataSeed),
    [dataSeed, dataset, noise],
  );
  const points = useMemo(() => [...basePoints, ...customPoints], [basePoints, customPoints]);
  const features = useMemo(() => points.map(({ x, y }) => [x, y]), [points]);

  const fitted = useMemo<FittedClustering | { error: string } | null>(() => {
    if (!mounted) return null;
    try {
      return fitClustering(activeAlgorithm, features, {
        eps,
        minSamples,
        minClusterSize,
        bandwidth,
        k,
        threshold,
        damping,
        preference,
        strategy,
        modelSeed,
      });
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'The model could not be fitted.' };
    }
  }, [activeAlgorithm, bandwidth, damping, eps, features, k, minClusterSize, minSamples, modelSeed, mounted, preference, strategy, threshold]);

  const result = fitted && 'error' in fitted ? null : fitted;
  const clusterLabels = result
    ? Array.from(new Set(result.labels.filter((label) => label >= 0))).sort((a, b) => a - b)
    : [];
  const noiseCount = result?.labels.filter((label) => label === -1).length ?? 0;
  const insight = result
    ? algorithmInsight(activeAlgorithm, clusterLabels.length, noiseCount, dataset)
    : '';

  const changeDataset = (next: Dataset) => {
    setDataset(next);
    setCustomPoints([]);
    setDataSeed((seed) => seed + 1);
  };

  const regenerate = () => {
    setDataSeed((seed) => seed + 1);
    setModelSeed((seed) => seed + 1);
    setCustomPoints([]);
  };

  const rerollModel = () => setModelSeed((seed) => seed + 1);

  const addCustomPoint = (x: number, y: number) => {
    const point = {
      x: Math.max(X_DOMAIN[0], Math.min(X_DOMAIN[1], x)),
      y: Math.max(Y_DOMAIN[0], Math.min(Y_DOMAIN[1], y)),
    };
    setCustomPoints((current) => [...current, point].slice(-24));
  };

  const handlePlotClick = (event: ReactMouseEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / bounds.width) * VIEW.width;
    const viewY = ((event.clientY - bounds.top) / bounds.height) * VIEW.height;
    const plotRight = VIEW.width - VIEW.right;
    const plotBottom = VIEW.height - VIEW.bottom;
    if (viewX < VIEW.left || viewX > plotRight || viewY < VIEW.top || viewY > plotBottom) return;
    const x = X_DOMAIN[0] + ((viewX - VIEW.left) / (plotRight - VIEW.left)) * (X_DOMAIN[1] - X_DOMAIN[0]);
    const y = Y_DOMAIN[1] - ((viewY - VIEW.top) / (plotBottom - VIEW.top)) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
    addCustomPoint(x, y);
  };

  return (
    <section className={styles.playground} aria-label={`${ALGORITHM_NAMES[activeAlgorithm]} interactive playground`}>
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}><span className={styles.liveDot} /> Live clustering · runs in your browser</div>
          <h2 className={styles.title}>Learn {ALGORITHM_NAMES[activeAlgorithm]} by changing it</h2>
          <p className={styles.subtitle}>{ALGORITHM_DESCRIPTIONS[activeAlgorithm]} Click the chart to add a point and refit.</p>
        </div>
        <code className={styles.modelBadge}>{ALGORITHM_BADGES[activeAlgorithm]}</code>
      </header>

      <div className={styles.controls}>
        {algorithm === 'advanced' && (
          <label className={styles.control}>
            <span className={styles.controlLabel}>Algorithm</span>
            <select
              className={styles.select}
              value={advancedAlgorithm}
              onChange={(event) => setAdvancedAlgorithm(event.target.value as AdvancedAlgorithm)}
            >
              <option value="birch">Birch</option>
              <option value="affinityPropagation">Affinity Propagation</option>
              <option value="bisectingKMeans">Bisecting K-Means</option>
            </select>
          </label>
        )}

        <label className={styles.control}>
          <span className={styles.controlLabel}>Dataset</span>
          <select className={styles.select} value={dataset} onChange={(event) => changeDataset(event.target.value as Dataset)}>
            <option value="blobs">Compact blobs</option>
            <option value="moons">Two moons</option>
            <option value="variableDensity">Variable density</option>
          </select>
        </label>

        <label className={styles.control} htmlFor={`${controlId}-noise`}>
          <span className={styles.controlLabel}>Data noise <span className={styles.controlValue}>{noise.toFixed(2)}</span></span>
          <input id={`${controlId}-noise`} className={styles.range} type="range" min="0.02" max="0.35" step="0.01" value={noise} onChange={(event) => setNoise(Number(event.target.value))} />
        </label>

        {(activeAlgorithm === 'dbscan' || activeAlgorithm === 'optics') && (
          <label className={styles.control} htmlFor={`${controlId}-eps`}>
            <span className={styles.controlLabel}>{activeAlgorithm === 'optics' ? 'Extraction eps' : 'Neighborhood eps'} <span className={styles.controlValue}>{eps.toFixed(2)}</span></span>
            <input id={`${controlId}-eps`} className={styles.range} type="range" min="0.16" max="1.05" step="0.01" value={eps} onChange={(event) => setEps(Number(event.target.value))} />
          </label>
        )}

        {(activeAlgorithm === 'dbscan' || activeAlgorithm === 'hdbscan' || activeAlgorithm === 'optics') && (
          <label className={styles.control} htmlFor={`${controlId}-min-samples`}>
            <span className={styles.controlLabel}>Min samples <span className={styles.controlValue}>{minSamples}</span></span>
            <input id={`${controlId}-min-samples`} className={styles.range} type="range" min="2" max="14" step="1" value={minSamples} onChange={(event) => setMinSamples(Number(event.target.value))} />
          </label>
        )}

        {activeAlgorithm === 'hdbscan' && (
          <label className={styles.control} htmlFor={`${controlId}-min-cluster`}>
            <span className={styles.controlLabel}>Min cluster size <span className={styles.controlValue}>{minClusterSize}</span></span>
            <input id={`${controlId}-min-cluster`} className={styles.range} type="range" min="2" max="18" step="1" value={minClusterSize} onChange={(event) => setMinClusterSize(Number(event.target.value))} />
          </label>
        )}

        {activeAlgorithm === 'meanShift' && (
          <label className={styles.control} htmlFor={`${controlId}-bandwidth`}>
            <span className={styles.controlLabel}>Bandwidth <span className={styles.controlValue}>{bandwidth.toFixed(2)}</span></span>
            <input id={`${controlId}-bandwidth`} className={styles.range} type="range" min="0.2" max="1.6" step="0.02" value={bandwidth} onChange={(event) => setBandwidth(Number(event.target.value))} />
          </label>
        )}

        {(activeAlgorithm === 'kmeansPlusPlus' || activeAlgorithm === 'birch' || activeAlgorithm === 'bisectingKMeans') && (
          <label className={styles.control} htmlFor={`${controlId}-clusters`}>
            <span className={styles.controlLabel}>{activeAlgorithm === 'kmeansPlusPlus' ? 'Seeds (k)' : 'Clusters'} <span className={styles.controlValue}>{k}</span></span>
            <input id={`${controlId}-clusters`} className={styles.range} type="range" min="2" max="6" step="1" value={k} onChange={(event) => setK(Number(event.target.value))} />
          </label>
        )}

        {activeAlgorithm === 'birch' && (
          <label className={styles.control} htmlFor={`${controlId}-threshold`}>
            <span className={styles.controlLabel}>CF threshold <span className={styles.controlValue}>{threshold.toFixed(2)}</span></span>
            <input id={`${controlId}-threshold`} className={styles.range} type="range" min="0.12" max="0.9" step="0.02" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
          </label>
        )}

        {activeAlgorithm === 'affinityPropagation' && (
          <>
            <label className={styles.control} htmlFor={`${controlId}-preference`}>
              <span className={styles.controlLabel}>Preference <span className={styles.controlValue}>{preference.toFixed(1)}</span></span>
              <input id={`${controlId}-preference`} className={styles.range} type="range" min="-8" max="-0.5" step="0.1" value={preference} onChange={(event) => setPreference(Number(event.target.value))} />
            </label>
            <label className={styles.control} htmlFor={`${controlId}-damping`}>
              <span className={styles.controlLabel}>Damping <span className={styles.controlValue}>{damping.toFixed(2)}</span></span>
              <input id={`${controlId}-damping`} className={styles.range} type="range" min="0.5" max="0.94" step="0.01" value={damping} onChange={(event) => setDamping(Number(event.target.value))} />
            </label>
          </>
        )}

        {activeAlgorithm === 'bisectingKMeans' && (
          <div className={styles.control}>
            <span className={styles.controlLabel}>Split strategy</span>
            <div className={styles.segment} role="group" aria-label="Bisecting K-Means split strategy">
              <button type="button" className={`${styles.segmentButton} ${strategy === 'biggestInertia' ? styles.segmentButtonActive : ''}`} aria-pressed={strategy === 'biggestInertia'} onClick={() => setStrategy('biggestInertia')}>Inertia</button>
              <button type="button" className={`${styles.segmentButton} ${strategy === 'largestCluster' ? styles.segmentButtonActive : ''}`} aria-pressed={strategy === 'largestCluster'} onClick={() => setStrategy('largestCluster')}>Size</button>
            </div>
          </div>
        )}
      </div>

      {!fitted && <div className={styles.loading}>Fitting the interactive model…</div>}
      {fitted && 'error' in fitted && <div className={styles.error}>Could not fit this clustering model: {fitted.error}</div>}
      {result && (
        <div className={styles.visualGrid}>
          <div className={styles.panel}>
            <div className={styles.metricGrid} aria-live="polite">
              <div className={styles.metric}><span>{activeAlgorithm === 'kmeansPlusPlus' ? 'Seed regions' : 'Clusters'}</span><strong>{clusterLabels.length}</strong></div>
              <div className={styles.metric}><span>{activeAlgorithm === 'kmeansPlusPlus' ? 'Seeds' : 'Noise'}</span><strong>{activeAlgorithm === 'kmeansPlusPlus' ? result.centers.length : noiseCount}</strong></div>
              <div className={styles.metric}><span>{result.detailLabel}</span><strong title={result.detailValue}>{result.detailValue}</strong></div>
            </div>

            <div className={styles.panelHeader}>
              <div>
                <h3 className={styles.panelTitle}>Live cluster map</h3>
                <p className={styles.panelHint}>
                  {activeAlgorithm === 'kmeansPlusPlus'
                    ? 'Colors are nearest-seed Voronoi regions; numbered observations are the selected seeds.'
                    : 'Colors are fitted labels; gray points are noise; outlined points were added by you.'}
                </p>
              </div>
            </div>

            <svg className={styles.chart} viewBox={`0 0 ${VIEW.width} ${VIEW.height}`} role="img" aria-label={`${ALGORITHM_NAMES[activeAlgorithm]} fitted clusters`} onClick={handlePlotClick}>
              <rect x={VIEW.left} y={VIEW.top} width={VIEW.width - VIEW.left - VIEW.right} height={VIEW.height - VIEW.top - VIEW.bottom} rx="8" className={styles.plotBackground} />
              {[-2, -1, 0, 1, 2].map((tick) => (
                <g key={`x-${tick}`} aria-hidden="true">
                  <line x1={xScale(tick)} x2={xScale(tick)} y1={VIEW.top} y2={VIEW.height - VIEW.bottom} className={styles.gridLine} />
                  <text x={xScale(tick)} y={VIEW.height - VIEW.bottom + 18} textAnchor="middle" className={styles.axisText}>{tick}</text>
                </g>
              ))}
              {[-2, -1, 0, 1, 2].map((tick) => (
                <g key={`y-${tick}`} aria-hidden="true">
                  <line x1={VIEW.left} x2={VIEW.width - VIEW.right} y1={yScale(tick)} y2={yScale(tick)} className={styles.gridLine} />
                  <text x={VIEW.left - 10} y={yScale(tick) + 3.5} textAnchor="end" className={styles.axisText}>{tick}</text>
                </g>
              ))}
              {points.map((point, index) => {
                const label = result.labels[index] ?? -1;
                const isCustom = index >= basePoints.length;
                const isSeed = result.seedIndices?.includes(index) ?? false;
                return (
                  <g key={`${index}-${point.x}-${point.y}`}>
                    <circle
                      cx={xScale(point.x)}
                      cy={yScale(point.y)}
                      r={isSeed ? 7 : isCustom ? 6 : 4.8}
                      fill={label === -1 ? NOISE_COLOR : clusterColor(label)}
                      fillOpacity={result.strengths ? 0.35 + (result.strengths[index] ?? 0) * 0.65 : 0.88}
                      stroke={isCustom || isSeed ? 'var(--cluster-ink)' : 'var(--cluster-paper)'}
                      strokeWidth={isCustom || isSeed ? 2.1 : 1.1}
                    />
                    {isSeed && (
                      <text x={xScale(point.x)} y={yScale(point.y) - 10} textAnchor="middle" className={styles.seedLabel}>
                        {(result.seedIndices?.indexOf(index) ?? 0) + 1}
                      </text>
                    )}
                  </g>
                );
              })}
              {result.centers.map((center, index) => {
                if (activeAlgorithm === 'kmeansPlusPlus') return null;
                const x = xScale(center[0]);
                const y = yScale(center[1]);
                const centerLabel = result.centerLabels?.[index] ?? index;
                return <path key={`${center[0]}-${center[1]}-${index}`} d={`M ${x} ${y - 7} L ${x + 7} ${y} L ${x} ${y + 7} L ${x - 7} ${y} Z`} fill={clusterColor(centerLabel)} stroke="var(--cluster-paper)" strokeWidth="2.4" />;
              })}
              <text x={(VIEW.left + VIEW.width - VIEW.right) / 2} y={VIEW.height - 7} textAnchor="middle" className={styles.axisText}>Feature 1</text>
              <text x="13" y={(VIEW.top + VIEW.height - VIEW.bottom) / 2} textAnchor="middle" transform={`rotate(-90 13 ${(VIEW.top + VIEW.height - VIEW.bottom) / 2})`} className={styles.axisText}>Feature 2</text>
            </svg>

            <div className={styles.actions}>
              <button type="button" className={styles.button} onClick={regenerate}>New data</button>
              {(activeAlgorithm === 'kmeansPlusPlus' || activeAlgorithm === 'affinityPropagation' || activeAlgorithm === 'bisectingKMeans') && <button type="button" className={styles.button} onClick={rerollModel}>Reroll model</button>}
              {customPoints.length > 0 && <button type="button" className={styles.button} onClick={() => setCustomPoints((current) => current.slice(0, -1))}>Undo point</button>}
              {customPoints.length > 0 && <button type="button" className={styles.button} onClick={() => setCustomPoints([])}>Clear points</button>}
            </div>
            <div className={styles.pointEntry} role="group" aria-label="Add an observation by coordinates">
              <span className={styles.pointEntryLabel}>Keyboard point</span>
              <label>
                <span>X</span>
                <input className={styles.numberInput} type="number" min={X_DOMAIN[0]} max={X_DOMAIN[1]} step="0.1" value={manualX} onChange={(event) => setManualX(Number(event.target.value))} />
              </label>
              <label>
                <span>Y</span>
                <input className={styles.numberInput} type="number" min={Y_DOMAIN[0]} max={Y_DOMAIN[1]} step="0.1" value={manualY} onChange={(event) => setManualY(Number(event.target.value))} />
              </label>
              <button type="button" className={styles.button} onClick={() => addCustomPoint(manualX, manualY)}>Add point</button>
            </div>
          </div>

          <aside className={styles.insight}>
            <span className={styles.insightLabel}>What to notice</span>
            <p>{insight}</p>
            <div className={styles.legend}>
              {clusterLabels.slice(0, 12).map((label) => <span key={label}><i style={{ background: clusterColor(label) }} />{activeAlgorithm === 'kmeansPlusPlus' ? 'Region' : 'Cluster'} {label + 1}</span>)}
              {clusterLabels.length > 12 && <span className={styles.legendMore}>+{clusterLabels.length - 12} more uniquely colored clusters</span>}
              {noiseCount > 0 && <span><i style={{ background: NOISE_COLOR }} />Noise</span>}
            </div>
          </aside>
        </div>
      )}
    </section>
  );
}
