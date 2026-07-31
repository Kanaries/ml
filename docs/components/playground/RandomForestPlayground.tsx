'use client';

import { useMemo, useState } from 'react';
import { Ensemble, Tree } from '@kanaries/ml';
import { COLORS, makeBlobs, makeMoons, makeXor, type Point2D } from './data';
import styles from './playground.module.css';

type Preset = 'moons' | 'xor' | 'blobs';

function accuracy(actual: number[], predicted: number[]) {
  return actual.reduce((sum, value, index) => sum + Number(value === predicted[index]), 0) / Math.max(1, actual.length);
}

export function RandomForestPlayground() {
  const [preset, setPreset] = useState<Preset>('moons');
  const [trees, setTrees] = useState(35);
  const [depth, setDepth] = useState(5);
  const [noise, setNoise] = useState(0.14);

  const points = useMemo(() => {
    if (preset === 'xor') return makeXor(90, 47).map((point, index) => ({ ...point, x: point.x + Math.sin(index) * noise, y: point.y + Math.cos(index * 1.7) * noise }));
    if (preset === 'blobs') return makeBlobs(90, [[-1.4, -1], [1.35, 1]], 0.42 + noise, 33);
    return makeMoons(90, noise, 23).map((point) => ({ ...point, x: point.x * 1.45 - 0.7, y: point.y * 1.55 }));
  }, [noise, preset]);

  const result = useMemo(() => {
    const train = points.filter((_, index) => index % 5 !== 0);
    const holdout = points.filter((_, index) => index % 5 === 0);
    const trainX = train.map((point) => [point.x, point.y]);
    const trainY = train.map((point) => point.label ?? 0);
    const holdoutX = holdout.map((point) => [point.x, point.y]);
    const holdoutY = holdout.map((point) => point.label ?? 0);
    const forest = new Ensemble.RandomForestClassifier({ nEstimators: trees, max_depth: depth, min_samples_split: 3, maxFeatures: 'sqrt', randomState: 42 });
    const single = new Tree.DecisionTreeClassifier({ max_depth: depth, min_samples_split: 3, criterion: 'gini', randomState: 42 });
    forest.fit(trainX, trainY);
    single.fit(trainX, trainY);
    const columns = 32;
    const rows = 26;
    const cells = Array.from({ length: columns * rows }, (_, index) => ({ column: index % columns, row: Math.floor(index / columns), x: -3 + (index % columns + 0.5) * 6 / columns, y: 3 - (Math.floor(index / columns) + 0.5) * 6 / rows }));
    const features = cells.map((cell) => [cell.x, cell.y]);
    const forestGrid = forest.predict(features);
    const treeGrid = single.predict(features);
    const disagreement = forestGrid.reduce((sum, value, index) => sum + Number(value !== treeGrid[index]), 0) / cells.length;
    return { cells, columns, rows, forestGrid, treeGrid, disagreement, forestAccuracy: accuracy(holdoutY, forest.predict(holdoutX)), treeAccuracy: accuracy(holdoutY, single.predict(holdoutX)) };
  }, [depth, points, trees]);

  const renderChart = (labels: number[], title: string) => (
    <div className={styles.card}>
      <h2 className={styles.chartTitle}>{title}</h2>
      <svg className={styles.svg} viewBox="0 0 600 400" role="img" aria-label={`${title} decision regions`}>
        {result.cells.map((cell, index) => <rect key={index} x={50 + cell.column * 500 / result.columns} y={25 + cell.row * 350 / result.rows} width={500 / result.columns + 0.5} height={350 / result.rows + 0.5} fill={COLORS[labels[index]]} opacity="0.15" />)}
        <rect x="50" y="25" width="500" height="350" fill="none" stroke="currentColor" opacity="0.35" />
        {points.map((point, index) => <circle key={index} cx={50 + (point.x + 3) / 6 * 500} cy={25 + (3 - point.y) / 6 * 350} r="4.2" fill={COLORS[point.label ?? 0]} stroke="white" strokeWidth="1.2" />)}
        <text x="300" y="396" textAnchor="middle" className={styles.axisText}>Feature 1</text><text x="13" y="200" textAnchor="middle" transform="rotate(-90 13 200)" className={styles.axisText}>Feature 2</text>
      </svg>
    </div>
  );

  return (
    <div className={styles.root} style={{ '--tool-accent': '#17605e' } as React.CSSProperties}>
      <div className={styles.controls} style={{ marginBottom: '0.8rem' }}>
        <div className={styles.chartGrid}>
          <div className={styles.controlGroup}><span className={styles.label}>Dataset</span><div className={styles.chips}>{(['moons', 'xor', 'blobs'] as Preset[]).map((name) => <button key={name} className={`${styles.chip} ${preset === name ? styles.chipActive : ''}`} type="button" onClick={() => setPreset(name)}>{name}</button>)}</div></div>
          <div className={styles.controlGroup}><label htmlFor="rf-trees">Trees in forest</label><div className={styles.rangeRow}><input id="rf-trees" type="range" min="5" max="75" step="5" value={trees} onChange={(event) => setTrees(Number(event.target.value))} /><output>{trees}</output></div></div>
          <div className={styles.controlGroup}><label htmlFor="rf-depth">Maximum depth</label><div className={styles.rangeRow}><input id="rf-depth" type="range" min="1" max="8" value={depth} onChange={(event) => setDepth(Number(event.target.value))} /><output>{depth}</output></div></div>
          <div className={styles.controlGroup}><label htmlFor="rf-noise">Dataset noise</label><div className={styles.rangeRow}><input id="rf-noise" type="range" min="0.02" max="0.38" step="0.02" value={noise} onChange={(event) => setNoise(Number(event.target.value))} /><output>{noise.toFixed(2)}</output></div></div>
        </div>
      </div>
      <div className={styles.metricGrid}>
        <div className={styles.metric}><span>Forest holdout</span><strong>{(result.forestAccuracy * 100).toFixed(1)}%</strong></div>
        <div className={styles.metric}><span>Single tree holdout</span><strong>{(result.treeAccuracy * 100).toFixed(1)}%</strong></div>
        <div className={styles.metric}><span>Boundary disagreement</span><strong>{(result.disagreement * 100).toFixed(1)}%</strong></div>
      </div>
      <div className={styles.chartGrid}>{renderChart(result.forestGrid, `Random Forest · ${trees} trees`)}{renderChart(result.treeGrid, 'One decision tree')}</div>
      <p className={styles.hint}>Both models use the same training split and maximum depth. The forest bootstraps rows and samples features per tree, then combines votes; the comparison isolates the effect of aggregation.</p>
    </div>
  );
}
