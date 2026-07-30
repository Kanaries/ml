'use client';

import {
  useEffect,
  useId,
  useMemo,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import { Tree } from '@kanaries/ml';
import styles from './decisionTreePlayground.module.css';

type TreeTask = 'classification' | 'regression';
type TreeVariant = 'decision' | 'extra';
type ClassificationDataset = 'moons' | 'xor' | 'rings';
type RegressionDataset = 'wave' | 'steps' | 'bump';
type Dataset = ClassificationDataset | RegressionDataset;

type Sample = {
  x: number;
  y: number;
  target: number;
  custom?: boolean;
};

type PredictiveModel = {
  fit: (features: number[][], targets: number[]) => void;
  predict: (features: number[][]) => number[];
};

type TreeNode = {
  splitIndex: number;
  nodeValue: number;
  y: number;
  leftChild: TreeNode | null;
  rightChild: TreeNode | null;
};

type TreeStats = {
  depth: number;
  leaves: number;
  nodes: number;
};

type FittedTree = {
  model: PredictiveModel;
  root: TreeNode | null;
  stats: TreeStats;
  trainScore: number;
  validationScore: number;
};

export type DecisionTreePlaygroundProps = {
  task: TreeTask;
  variant?: TreeVariant;
};

const VIEW = {
  width: 640,
  height: 390,
  left: 52,
  right: 18,
  top: 18,
  bottom: 42,
};

const X_DOMAIN: [number, number] = [-2.8, 2.8];
const Y_DOMAIN: [number, number] = [-2.35, 2.35];

const CLASSIFICATION_DATASETS: Array<{ value: ClassificationDataset; label: string }> = [
  { value: 'moons', label: 'Two moons' },
  { value: 'xor', label: 'XOR quadrants' },
  { value: 'rings', label: 'Nested rings' },
];

const REGRESSION_DATASETS: Array<{ value: RegressionDataset; label: string }> = [
  { value: 'wave', label: 'Noisy wave' },
  { value: 'steps', label: 'Threshold steps' },
  { value: 'bump', label: 'Central bump' },
];

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

function generateClassificationSamples(
  dataset: ClassificationDataset,
  count: number,
  noise: number,
  seed: number,
): Sample[] {
  const random = createRandom(seed);
  const samples: Sample[] = [];

  if (dataset === 'moons') {
    const half = Math.floor(count / 2);
    for (let index = 0; index < count; index += 1) {
      const target = index < half ? 0 : 1;
      const t = random() * Math.PI;
      const jitterX = gaussian(random) * noise;
      const jitterY = gaussian(random) * noise;
      if (target === 0) {
        samples.push({
          x: (Math.cos(t) - 0.45) * 1.28 + jitterX,
          y: (Math.sin(t) - 0.28) * 1.28 + jitterY,
          target,
        });
      } else {
        samples.push({
          x: (0.58 - Math.cos(t)) * 1.28 + jitterX,
          y: (-Math.sin(t) + 0.42) * 1.28 + jitterY,
          target,
        });
      }
    }
    return samples;
  }

  if (dataset === 'xor') {
    for (let index = 0; index < count; index += 1) {
      const x = random() * 4.7 - 2.35;
      const y = random() * 4 - 2;
      const noisyX = x + gaussian(random) * noise * 0.45;
      const noisyY = y + gaussian(random) * noise * 0.45;
      const target = noisyX * noisyY >= 0 ? 1 : 0;
      samples.push({ x: noisyX, y: noisyY, target });
    }
    return samples;
  }

  for (let index = 0; index < count; index += 1) {
    const target = index % 2;
    const angle = random() * Math.PI * 2;
    const baseRadius = target === 0 ? 0.72 : 1.72;
    const radius = baseRadius + gaussian(random) * (noise + 0.05);
    samples.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      target,
    });
  }
  return samples;
}

function regressionSignal(dataset: RegressionDataset, x: number) {
  if (dataset === 'steps') {
    if (x < -0.9) return -1.15;
    if (x < 0.75) return 0.25;
    return 1.35;
  }
  if (dataset === 'bump') {
    return 2.05 * Math.exp(-0.85 * x * x) - 0.78 + 0.12 * x;
  }
  return Math.sin(x * 1.55) * 0.95 + x * 0.18;
}

function generateRegressionSamples(
  dataset: RegressionDataset,
  count: number,
  noise: number,
  seed: number,
): Sample[] {
  const random = createRandom(seed);
  return Array.from({ length: count }, () => {
    const x = random() * 5.1 - 2.55;
    const y = regressionSignal(dataset, x) + gaussian(random) * noise;
    return { x, y, target: y };
  }).sort((a, b) => a.x - b.x);
}

function featureRows(task: TreeTask, samples: Sample[]) {
  return samples.map((sample) => (task === 'classification' ? [sample.x, sample.y] : [sample.x]));
}

function accuracy(actual: number[], predicted: number[]) {
  if (actual.length === 0) return 0;
  const correct = actual.reduce((total, value, index) => total + Number(value === predicted[index]), 0);
  return correct / actual.length;
}

function rmse(actual: number[], predicted: number[]) {
  if (actual.length === 0) return 0;
  const squaredError = actual.reduce((total, value, index) => total + (value - predicted[index]) ** 2, 0);
  return Math.sqrt(squaredError / actual.length);
}

function inspectTree(root: TreeNode | null): TreeStats {
  if (!root) return { depth: 0, leaves: 0, nodes: 0 };
  if (root.splitIndex === -1 || !root.leftChild || !root.rightChild) {
    return { depth: 0, leaves: 1, nodes: 1 };
  }
  const left = inspectTree(root.leftChild);
  const right = inspectTree(root.rightChild);
  return {
    depth: 1 + Math.max(left.depth, right.depth),
    leaves: left.leaves + right.leaves,
    nodes: 1 + left.nodes + right.nodes,
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

function xFromView(value: number) {
  const plotWidth = VIEW.width - VIEW.left - VIEW.right;
  return X_DOMAIN[0] + ((value - VIEW.left) / plotWidth) * (X_DOMAIN[1] - X_DOMAIN[0]);
}

function yFromView(value: number) {
  const plotHeight = VIEW.height - VIEW.top - VIEW.bottom;
  return Y_DOMAIN[1] - ((value - VIEW.top) / plotHeight) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
}

function PlotAxes({ xLabel, yLabel }: { xLabel: string; yLabel: string }) {
  const xTicks = [-2, -1, 0, 1, 2];
  const yTicks = [-2, -1, 0, 1, 2];
  const plotRight = VIEW.width - VIEW.right;
  const plotBottom = VIEW.height - VIEW.bottom;

  return (
    <g aria-hidden="true">
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line
            x1={xScale(tick)}
            x2={xScale(tick)}
            y1={VIEW.top}
            y2={plotBottom}
            stroke="var(--tree-grid)"
          />
          <text x={xScale(tick)} y={plotBottom + 18} textAnchor="middle" fill="var(--tree-muted)" fontSize="10">
            {tick}
          </text>
        </g>
      ))}
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line
            x1={VIEW.left}
            x2={plotRight}
            y1={yScale(tick)}
            y2={yScale(tick)}
            stroke="var(--tree-grid)"
          />
          <text x={VIEW.left - 10} y={yScale(tick) + 3.5} textAnchor="end" fill="var(--tree-muted)" fontSize="10">
            {tick}
          </text>
        </g>
      ))}
      <line x1={VIEW.left} x2={plotRight} y1={plotBottom} y2={plotBottom} stroke="var(--tree-line)" />
      <line x1={VIEW.left} x2={VIEW.left} y1={VIEW.top} y2={plotBottom} stroke="var(--tree-line)" />
      <text x={(VIEW.left + plotRight) / 2} y={VIEW.height - 7} textAnchor="middle" fill="var(--tree-muted)" fontSize="11">
        {xLabel}
      </text>
      <text
        x="13"
        y={(VIEW.top + plotBottom) / 2}
        textAnchor="middle"
        fill="var(--tree-muted)"
        fontSize="11"
        transform={`rotate(-90 13 ${(VIEW.top + plotBottom) / 2})`}
      >
        {yLabel}
      </text>
    </g>
  );
}

function ClassificationPlot({
  model,
  samples,
  clipId,
  onClick,
}: {
  model: PredictiveModel;
  samples: Sample[];
  clipId: string;
  onClick: (event: ReactMouseEvent<SVGSVGElement>) => void;
}) {
  const regions = useMemo(() => {
    const columns = 54;
    const rows = 42;
    const features: number[][] = [];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const x = X_DOMAIN[0] + ((column + 0.5) / columns) * (X_DOMAIN[1] - X_DOMAIN[0]);
        const y = Y_DOMAIN[1] - ((row + 0.5) / rows) * (Y_DOMAIN[1] - Y_DOMAIN[0]);
        features.push([x, y]);
      }
    }
    const predictions = model.predict(features);
    const plotWidth = VIEW.width - VIEW.left - VIEW.right;
    const plotHeight = VIEW.height - VIEW.top - VIEW.bottom;
    const cellWidth = plotWidth / columns + 0.45;
    const cellHeight = plotHeight / rows + 0.45;
    const paths = ['', ''];
    predictions.forEach((prediction, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = VIEW.left + (column / columns) * plotWidth;
      const y = VIEW.top + (row / rows) * plotHeight;
      const classIndex = prediction === 1 ? 1 : 0;
      paths[classIndex] += `M${x.toFixed(2)} ${y.toFixed(2)}h${cellWidth.toFixed(2)}v${cellHeight.toFixed(2)}h-${cellWidth.toFixed(2)}Z`;
    });
    return paths;
  }, [model]);

  return (
    <svg
      className={styles.chart}
      viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
      role="img"
      aria-label="Interactive decision boundary. Click inside the plot to add a labeled observation."
      onClick={onClick}
    >
      <defs>
        <clipPath id={clipId}>
          <rect
            x={VIEW.left}
            y={VIEW.top}
            width={VIEW.width - VIEW.left - VIEW.right}
            height={VIEW.height - VIEW.top - VIEW.bottom}
          />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <path d={regions[0]} fill="var(--tree-coral-soft)" />
        <path d={regions[1]} fill="var(--tree-blue-soft)" />
      </g>
      <PlotAxes xLabel="feature x₁" yLabel="feature x₂" />
      <g clipPath={`url(#${clipId})`}>
        {samples.map((sample, index) => (
          <g key={`${sample.custom ? 'custom' : 'base'}-${index}-${sample.x.toFixed(3)}`}>
            {sample.custom && (
              <circle
                cx={xScale(sample.x)}
                cy={yScale(sample.y)}
                r="8"
                fill="none"
                stroke="var(--tree-ink)"
                strokeWidth="1.4"
                opacity="0.8"
              />
            )}
            <circle
              cx={xScale(sample.x)}
              cy={yScale(sample.y)}
              r={sample.custom ? 5.1 : 4.2}
              fill={sample.target === 1 ? 'var(--tree-blue)' : 'var(--tree-coral)'}
              stroke="var(--tree-paper)"
              strokeWidth="1.25"
            >
              <title>{`Class ${sample.target === 1 ? 'B' : 'A'} · x₁ ${sample.x.toFixed(2)}, x₂ ${sample.y.toFixed(2)}`}</title>
            </circle>
          </g>
        ))}
      </g>
    </svg>
  );
}

function RegressionPlot({
  model,
  dataset,
  samples,
  clipId,
  onClick,
}: {
  model: PredictiveModel;
  dataset: RegressionDataset;
  samples: Sample[];
  clipId: string;
  onClick: (event: ReactMouseEvent<SVGSVGElement>) => void;
}) {
  const curves = useMemo(() => {
    const count = 220;
    const xs = Array.from(
      { length: count },
      (_, index) => X_DOMAIN[0] + (index / (count - 1)) * (X_DOMAIN[1] - X_DOMAIN[0]),
    );
    const predictions = model.predict(xs.map((x) => [x]));
    const truePath = xs
      .map((x, index) => `${index === 0 ? 'M' : 'L'}${xScale(x).toFixed(2)} ${yScale(regressionSignal(dataset, x)).toFixed(2)}`)
      .join(' ');
    let predictionPath = `M${xScale(xs[0]).toFixed(2)} ${yScale(predictions[0]).toFixed(2)}`;
    for (let index = 1; index < xs.length; index += 1) {
      const x = xScale(xs[index]).toFixed(2);
      const previousY = yScale(predictions[index - 1]).toFixed(2);
      const y = yScale(predictions[index]).toFixed(2);
      predictionPath += ` L${x} ${previousY} L${x} ${y}`;
    }
    return { predictionPath, truePath };
  }, [dataset, model]);

  return (
    <svg
      className={styles.chart}
      viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
      role="img"
      aria-label="Interactive regression fit. Click inside the plot to add a numeric observation."
      onClick={onClick}
    >
      <defs>
        <clipPath id={clipId}>
          <rect
            x={VIEW.left}
            y={VIEW.top}
            width={VIEW.width - VIEW.left - VIEW.right}
            height={VIEW.height - VIEW.top - VIEW.bottom}
          />
        </clipPath>
      </defs>
      <PlotAxes xLabel="feature x" yLabel="target y" />
      <g clipPath={`url(#${clipId})`}>
        <path
          d={curves.truePath}
          fill="none"
          stroke="var(--tree-gold)"
          strokeWidth="2"
          strokeDasharray="6 5"
          opacity="0.8"
        />
        <path
          d={curves.predictionPath}
          fill="none"
          stroke="var(--tree-blue)"
          strokeWidth="3"
          strokeLinejoin="round"
        />
        {samples.map((sample, index) => (
          <g key={`${sample.custom ? 'custom' : 'base'}-${index}-${sample.x.toFixed(3)}`}>
            {sample.custom && (
              <circle
                cx={xScale(sample.x)}
                cy={yScale(sample.y)}
                r="8"
                fill="none"
                stroke="var(--tree-ink)"
                strokeWidth="1.4"
              />
            )}
            <circle
              cx={xScale(sample.x)}
              cy={yScale(sample.y)}
              r={sample.custom ? 5.1 : 4}
              fill="var(--tree-coral)"
              stroke="var(--tree-paper)"
              strokeWidth="1.2"
            >
              <title>{`x ${sample.x.toFixed(2)} · y ${sample.y.toFixed(2)}`}</title>
            </circle>
          </g>
        ))}
      </g>
    </svg>
  );
}

type LayoutNode = {
  id: string;
  node: TreeNode;
  depth: number;
  x: number;
  y: number;
  sampleCount: number;
  truncated: boolean;
};

type LayoutLink = {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

function buildTreeLayout(
  root: TreeNode,
  features: number[][],
  variant: TreeVariant,
): { nodes: LayoutNode[]; links: LayoutLink[] } {
  const nodes: LayoutNode[] = [];
  const links: LayoutLink[] = [];
  const maxVisibleDepth = 3;

  const walk = (
    node: TreeNode,
    indices: number[],
    depth: number,
    minX: number,
    maxX: number,
    id: string,
    parent?: { x: number; y: number },
  ) => {
    const x = (minX + maxX) / 2;
    const y = 38 + depth * 70;
    const isBranch = node.splitIndex !== -1 && Boolean(node.leftChild && node.rightChild);
    const truncated = isBranch && depth === maxVisibleDepth;
    nodes.push({ id, node, depth, x, y, sampleCount: indices.length, truncated });
    if (parent) links.push({ id: `link-${id}`, x1: parent.x, y1: parent.y, x2: x, y2: y });
    if (!isBranch || truncated || !node.leftChild || !node.rightChild) return;

    const leftIndices: number[] = [];
    const rightIndices: number[] = [];
    indices.forEach((index) => {
      const value = features[index][node.splitIndex];
      const goesLeft = variant === 'extra' ? value < node.nodeValue : value <= node.nodeValue;
      (goesLeft ? leftIndices : rightIndices).push(index);
    });
    const midpoint = (minX + maxX) / 2;
    walk(node.leftChild, leftIndices, depth + 1, minX, midpoint, `${id}L`, { x, y });
    walk(node.rightChild, rightIndices, depth + 1, midpoint, maxX, `${id}R`, { x, y });
  };

  walk(root, features.map((_, index) => index), 0, 20, 620, 'root');
  return { nodes, links };
}

function TreeDiagram({
  root,
  task,
  variant,
  features,
}: {
  root: TreeNode;
  task: TreeTask;
  variant: TreeVariant;
  features: number[][];
}) {
  const layout = useMemo(() => buildTreeLayout(root, features, variant), [features, root, variant]);
  const operator = variant === 'extra' ? '<' : '≤';

  return (
    <svg
      className={styles.treeChart}
      viewBox="0 0 640 288"
      role="img"
      aria-label="The fitted tree's first four levels"
    >
      {layout.links.map((link) => (
        <path
          key={link.id}
          d={`M${link.x1} ${link.y1 + 20} C${link.x1} ${(link.y1 + link.y2) / 2}, ${link.x2} ${(link.y1 + link.y2) / 2}, ${link.x2} ${link.y2 - 20}`}
          fill="none"
          stroke="var(--tree-line)"
          strokeWidth="1.5"
        />
      ))}
      {layout.nodes.map(({ id, node, depth, x, y, sampleCount, truncated }) => {
        const isLeaf = node.splitIndex === -1 || !node.leftChild || !node.rightChild;
        const width = depth >= 3 ? 66 : depth === 2 ? 78 : 92;
        const fill = isLeaf
          ? task === 'classification'
            ? node.y === 1
              ? 'var(--tree-blue-soft)'
              : 'var(--tree-coral-soft)'
            : node.y >= 0
              ? 'var(--tree-blue-soft)'
              : 'var(--tree-coral-soft)'
          : 'var(--tree-paper)';
        const stroke = isLeaf
          ? task === 'classification' && node.y !== 1
            ? 'var(--tree-coral)'
            : 'var(--tree-blue)'
          : 'var(--tree-line)';
        const primary = isLeaf
          ? task === 'classification'
            ? `Class ${node.y === 1 ? 'B' : 'A'}`
            : `ŷ ${node.y.toFixed(2)}`
          : `x${node.splitIndex + 1} ${operator} ${node.nodeValue.toFixed(2)}`;
        const secondary = truncated ? `${sampleCount} rows · …` : `${sampleCount} row${sampleCount === 1 ? '' : 's'}`;

        return (
          <g key={id}>
            <rect
              x={x - width / 2}
              y={y - 21}
              width={width}
              height="42"
              rx="9"
              fill={fill}
              stroke={stroke}
              strokeWidth="1.25"
            />
            <text x={x} y={y - 2} textAnchor="middle" fill="var(--tree-ink)" fontSize={depth >= 3 ? 9 : 10.5} fontWeight="700">
              {primary}
            </text>
            <text x={x} y={y + 12} textAnchor="middle" fill="var(--tree-muted)" fontSize={depth >= 3 ? 7.5 : 8.5}>
              {secondary}
            </text>
            <title>{isLeaf ? `${primary}, trained on ${sampleCount} rows` : `${primary}, ${sampleCount} rows reach this node`}</title>
          </g>
        );
      })}
    </svg>
  );
}

function modelName(task: TreeTask, variant: TreeVariant) {
  if (task === 'classification') {
    return variant === 'extra' ? 'ExtraTreeClassifier' : 'DecisionTreeClassifier';
  }
  return variant === 'extra' ? 'ExtraTreeRegressor' : 'DecisionTreeRegressor';
}

function learningInsight(
  task: TreeTask,
  variant: TreeVariant,
  maxDepth: number,
  minSamples: number,
  trainScore: number,
  validationScore: number,
) {
  if (maxDepth <= 2) {
    return {
      title: 'A shallow tree makes broad, legible rules.',
      text: task === 'classification'
        ? 'Increase max depth and watch one large color region break into smaller rectangles. That flexibility can fix underfitting, but it can also chase noisy points.'
        : 'Increase max depth and watch the prediction line gain more steps. Each new split narrows the interval represented by a leaf.',
    };
  }

  const gap = task === 'classification' ? trainScore - validationScore : validationScore - trainScore;
  if (maxDepth >= 6 && minSamples <= 4 && gap > (task === 'classification' ? 0.07 : 0.09)) {
    return {
      title: 'The tree is starting to memorize local detail.',
      text: 'Training performance is pulling away from the holdout score. Raise minimum samples per split or reduce depth to merge brittle regions back into stronger rules.',
    };
  }

  if (variant === 'extra') {
    return {
      title: 'Random thresholds trade precision for diversity.',
      text: 'Use “Reroll splits” with the same data and parameters. The tree changes because Extra Tree samples candidate thresholds instead of always choosing the globally best one.',
    };
  }

  return {
    title: 'Read the plot and the tree together.',
    text: 'Every internal node in the diagram adds an axis-aligned cut to the plot. Increase minimum samples per split to prevent small groups of observations from creating their own branch.',
  };
}

export function DecisionTreePlayground({
  task,
  variant = 'decision',
}: DecisionTreePlaygroundProps) {
  const [mounted, setMounted] = useState(false);
  const [dataset, setDataset] = useState<Dataset>(task === 'classification' ? 'moons' : 'wave');
  const [maxDepth, setMaxDepth] = useState(3);
  const [minSamples, setMinSamples] = useState(4);
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini');
  const [noise, setNoise] = useState(task === 'classification' ? 0.16 : 0.2);
  const [dataSeed, setDataSeed] = useState(17);
  const [modelSeed, setModelSeed] = useState(31);
  const [activeClass, setActiveClass] = useState<0 | 1>(1);
  const [customSamples, setCustomSamples] = useState<Sample[]>([]);
  const rawId = useId();
  const clipId = `tree-plot-${rawId.replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const name = modelName(task, variant);

  useEffect(() => setMounted(true), []);

  const baseSamples = useMemo(() => {
    if (task === 'classification') {
      return generateClassificationSamples(dataset as ClassificationDataset, 88, noise, dataSeed);
    }
    return generateRegressionSamples(dataset as RegressionDataset, 68, noise, dataSeed);
  }, [dataSeed, dataset, noise, task]);

  const validationSamples = useMemo(() => {
    if (task === 'classification') {
      return generateClassificationSamples(dataset as ClassificationDataset, 220, noise, dataSeed + 1009);
    }
    return generateRegressionSamples(dataset as RegressionDataset, 180, noise, dataSeed + 1009);
  }, [dataSeed, dataset, noise, task]);

  const samples = useMemo(() => [...baseSamples, ...customSamples], [baseSamples, customSamples]);
  const features = useMemo(() => featureRows(task, samples), [samples, task]);

  const fitted = useMemo<FittedTree | { error: string } | null>(() => {
    if (!mounted) return null;
    try {
      const commonOptions = {
        max_depth: maxDepth,
        min_samples_split: minSamples,
        randomState: modelSeed,
      };
      let model: PredictiveModel;
      if (task === 'classification') {
        const options = { ...commonOptions, criterion, max_features: 2 };
        model = variant === 'extra'
          ? new Tree.ExtraTreeClassifier(options)
          : new Tree.DecisionTreeClassifier(options);
      } else {
        const options = commonOptions;
        model = variant === 'extra'
          ? new Tree.ExtraTreeRegressor(options)
          : new Tree.DecisionTreeRegressor(options);
      }

      const targets = samples.map((sample) => sample.target);
      model.fit(features, targets);
      const trainPredictions = model.predict(features);
      const validationFeatures = featureRows(task, validationSamples);
      const validationTargets = validationSamples.map((sample) => sample.target);
      const validationPredictions = model.predict(validationFeatures);
      const internal = model as PredictiveModel & { dtree?: TreeNode | null; regTree?: TreeNode | null };
      const root = task === 'classification' ? internal.dtree ?? null : internal.regTree ?? null;

      return {
        model,
        root,
        stats: inspectTree(root),
        trainScore: task === 'classification'
          ? accuracy(targets, trainPredictions)
          : rmse(targets, trainPredictions),
        validationScore: task === 'classification'
          ? accuracy(validationTargets, validationPredictions)
          : rmse(validationTargets, validationPredictions),
      };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'The model could not be fitted.' };
    }
  }, [criterion, features, maxDepth, minSamples, modelSeed, mounted, samples, task, validationSamples, variant]);

  const handlePlotClick = (event: ReactMouseEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / bounds.width) * VIEW.width;
    const viewY = ((event.clientY - bounds.top) / bounds.height) * VIEW.height;
    const plotRight = VIEW.width - VIEW.right;
    const plotBottom = VIEW.height - VIEW.bottom;
    if (viewX < VIEW.left || viewX > plotRight || viewY < VIEW.top || viewY > plotBottom) return;
    const x = xFromView(viewX);
    const y = yFromView(viewY);
    setCustomSamples((current) => [
      ...current,
      task === 'classification'
        ? { x, y, target: activeClass, custom: true }
        : { x, y, target: y, custom: true },
    ].slice(-30));
  };

  const chooseDataset = (nextDataset: Dataset) => {
    setDataset(nextDataset);
    setCustomSamples([]);
    setModelSeed((seed) => seed + 1);
  };

  const regenerateData = () => {
    setDataSeed((seed) => seed + 1);
    setModelSeed((seed) => seed + 1);
    setCustomSamples([]);
  };

  const description = task === 'classification'
    ? variant === 'extra'
      ? 'Reroll randomized thresholds, then compare how depth changes the decision regions.'
      : 'Tune the tree and watch each split carve a new rectangle into feature space.'
    : variant === 'extra'
      ? 'Explore how randomized splits turn noisy observations into a piecewise prediction.'
      : 'Tune the tree and watch each leaf become one step in the fitted function.';

  const datasetOptions = task === 'classification' ? CLASSIFICATION_DATASETS : REGRESSION_DATASETS;
  const result = fitted && 'error' in fitted ? null : fitted;
  const insight = result
    ? learningInsight(task, variant, maxDepth, minSamples, result.trainScore, result.validationScore)
    : null;

  return (
    <section className={styles.playground} aria-label={`${name} interactive playground`}>
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}><span className={styles.liveDot} /> Live model · runs in your browser</div>
          <h2 className={styles.title}>Learn {name} by changing it</h2>
          <p className={styles.subtitle}>{description} Click the plot to add your own observation and refit instantly.</p>
        </div>
        <code className={styles.modelBadge}>Tree.{name}</code>
      </header>

      <div className={styles.controls}>
        <label className={styles.control}>
          <span className={styles.controlLabel}>Dataset</span>
          <select
            className={styles.select}
            value={dataset}
            onChange={(event) => chooseDataset(event.target.value as Dataset)}
          >
            {datasetOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>

        <label className={styles.control}>
          <span className={styles.controlLabel}>Max depth <span className={styles.controlValue}>{maxDepth}</span></span>
          <input
            className={styles.range}
            type="range"
            min="1"
            max="7"
            step="1"
            value={maxDepth}
            onChange={(event) => setMaxDepth(Number(event.target.value))}
          />
        </label>

        <label className={styles.control}>
          <span className={styles.controlLabel}>Min split <span className={styles.controlValue}>{minSamples}</span></span>
          <input
            className={styles.range}
            type="range"
            min="2"
            max="20"
            step="1"
            value={minSamples}
            onChange={(event) => setMinSamples(Number(event.target.value))}
          />
        </label>

        <label className={styles.control}>
          <span className={styles.controlLabel}>Noise <span className={styles.controlValue}>{noise.toFixed(2)}</span></span>
          <input
            className={styles.range}
            type="range"
            min="0"
            max={task === 'classification' ? '0.42' : '0.55'}
            step="0.01"
            value={noise}
            onChange={(event) => setNoise(Number(event.target.value))}
          />
        </label>

        {task === 'classification' && (
          <div className={styles.control}>
            <span className={styles.controlLabel}>Criterion</span>
            <div className={styles.segment} role="group" aria-label="Split criterion">
              {(['gini', 'entropy'] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`${styles.segmentButton} ${criterion === value ? styles.segmentButtonActive : ''}`}
                  aria-pressed={criterion === value}
                  onClick={() => setCriterion(value)}
                >
                  {value === 'gini' ? 'Gini' : 'Entropy'}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {!fitted && <div className={styles.loading}>Growing the interactive tree…</div>}
      {fitted && 'error' in fitted && <div className={styles.error}>Could not fit this tree: {fitted.error}</div>}
      {result && (
        <>
          <div className={styles.visualGrid}>
            <div className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <h3 className={styles.panelTitle}>{task === 'classification' ? 'Decision surface' : 'Piecewise fit'}</h3>
                  <p className={styles.panelHint}>
                    {task === 'classification'
                      ? 'Background color is the predicted class; outlined points are yours.'
                      : 'Blue is the model; the dashed gold line is the hidden signal.'}
                  </p>
                </div>
              </div>

              {task === 'classification' ? (
                <ClassificationPlot
                  model={result.model}
                  samples={samples}
                  clipId={clipId}
                  onClick={handlePlotClick}
                />
              ) : (
                <RegressionPlot
                  model={result.model}
                  dataset={dataset as RegressionDataset}
                  samples={samples}
                  clipId={clipId}
                  onClick={handlePlotClick}
                />
              )}

              <div className={styles.actions}>
                <button type="button" className={styles.button} onClick={regenerateData}>New data</button>
                {variant === 'extra' && (
                  <button type="button" className={styles.button} onClick={() => setModelSeed((seed) => seed + 1)}>
                    Reroll splits
                  </button>
                )}
                {customSamples.length > 0 && (
                  <button
                    type="button"
                    className={styles.button}
                    onClick={() => setCustomSamples((current) => current.slice(0, -1))}
                  >
                    Undo point
                  </button>
                )}
                {task === 'classification' && (
                  <div className={styles.classPicker}>
                    Add class
                    {([0, 1] as const).map((value) => (
                      <button
                        key={value}
                        type="button"
                        data-class={value}
                        className={`${styles.classButton} ${activeClass === value ? styles.classButtonActive : ''}`}
                        aria-label={`Add class ${value === 0 ? 'A' : 'B'} points`}
                        aria-pressed={activeClass === value}
                        onClick={() => setActiveClass(value)}
                      >
                        {value === 0 ? 'A' : 'B'}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <h3 className={styles.panelTitle}>What the tree learned</h3>
                  <p className={styles.panelHint}>
                    First four levels · actual depth {result.stats.depth} · {result.stats.nodes} nodes
                  </p>
                </div>
              </div>

              {result.root ? (
                <TreeDiagram root={result.root} task={task} variant={variant} features={features} />
              ) : (
                <div className={styles.loading}>Tree structure is not exposed by this package build.</div>
              )}

              <div className={styles.metrics}>
                <div className={styles.metric}>
                  <span className={styles.metricValue}>
                    {task === 'classification' ? `${(result.trainScore * 100).toFixed(1)}%` : result.trainScore.toFixed(3)}
                  </span>
                  <span className={styles.metricLabel}>{task === 'classification' ? 'Train accuracy' : 'Train RMSE'}</span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricValue}>
                    {task === 'classification' ? `${(result.validationScore * 100).toFixed(1)}%` : result.validationScore.toFixed(3)}
                  </span>
                  <span className={styles.metricLabel}>{task === 'classification' ? 'Holdout accuracy' : 'Holdout RMSE'}</span>
                </div>
                <div className={styles.metric}>
                  <span className={styles.metricValue}>{result.stats.leaves}</span>
                  <span className={styles.metricLabel}>Leaves</span>
                </div>
              </div>
            </div>
          </div>

          {insight && (
            <div className={styles.insight}>
              <span className={styles.insightIndex}>i</span>
              <div>
                <strong>{insight.title}</strong>
                <p>{insight.text}</p>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
