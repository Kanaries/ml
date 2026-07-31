'use client';

import { useEffect, useMemo, useState } from 'react';
import { KMath } from '@kanaries/ml';
import { CodeTabs } from '@/components/tools/CodeTabs';
import styles from './playground.module.css';

type ObjectiveName = 'bowl' | 'rosenbrock' | 'rippled';
type Vector = [number, number];
type PathPoint = { x: number; y: number; value: number };

const OPTIMIZERS = [
  { key: 'sgd', label: 'SGD', color: '#2b7a78' },
  { key: 'momentum', label: 'Momentum', color: '#d45d4c' },
  { key: 'adam', label: 'Adam', color: '#7b61a8' },
] as const;

function objective(name: ObjectiveName, [x, y]: Vector): { value: number; gradient: Vector } {
  if (name === 'rosenbrock') {
    const value = KMath.sum([(1 - x) ** 2, 5 * (y - x * x) ** 2]);
    return { value, gradient: [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)] };
  }
  if (name === 'rippled') {
    const value = KMath.sum([0.18 * x * x, 0.18 * y * y, 0.8 * Math.sin(1.8 * x) * Math.sin(1.8 * y)]);
    return { value, gradient: [0.36 * x + 1.44 * Math.cos(1.8 * x) * Math.sin(1.8 * y), 0.36 * y + 1.44 * Math.sin(1.8 * x) * Math.cos(1.8 * y)] };
  }
  return { value: KMath.sum([x * x, 0.45 * y * y]), gradient: [2 * x, 0.9 * y] };
}

function clipped([x, y]: Vector): Vector {
  const norm = Math.hypot(x, y);
  const scale = norm > 8 ? 8 / norm : 1;
  return [x * scale, y * scale];
}

function trajectories(name: ObjectiveName, learningRate: number, start: Vector, steps = 60) {
  return Object.fromEntries(OPTIMIZERS.map(({ key }) => {
    let position: Vector = [...start];
    let velocity: Vector = [0, 0];
    let first: Vector = [0, 0];
    let second: Vector = [0, 0];
    const path: PathPoint[] = [];
    for (let iteration = 0; iteration <= steps; iteration += 1) {
      const current = objective(name, position);
      path.push({ x: position[0], y: position[1], value: current.value });
      if (iteration === steps) break;
      const gradient = clipped(current.gradient);
      if (key === 'momentum') {
        velocity = [0.88 * velocity[0] + gradient[0], 0.88 * velocity[1] + gradient[1]];
        position = [position[0] - learningRate * velocity[0], position[1] - learningRate * velocity[1]];
      } else if (key === 'adam') {
        first = [0.9 * first[0] + 0.1 * gradient[0], 0.9 * first[1] + 0.1 * gradient[1]];
        second = [0.999 * second[0] + 0.001 * gradient[0] ** 2, 0.999 * second[1] + 0.001 * gradient[1] ** 2];
        const t = iteration + 1;
        const mHat: Vector = [first[0] / (1 - 0.9 ** t), first[1] / (1 - 0.9 ** t)];
        const vHat: Vector = [second[0] / (1 - 0.999 ** t), second[1] / (1 - 0.999 ** t)];
        position = [position[0] - learningRate * mHat[0] / (Math.sqrt(vHat[0]) + 1e-8), position[1] - learningRate * mHat[1] / (Math.sqrt(vHat[1]) + 1e-8)];
      } else {
        position = [position[0] - learningRate * gradient[0], position[1] - learningRate * gradient[1]];
      }
      position = [Math.max(-2.5, Math.min(2.5, position[0])), Math.max(-2.5, Math.min(2.5, position[1]))];
    }
    return [key, path];
  })) as Record<(typeof OPTIMIZERS)[number]['key'], PathPoint[]>;
}

const javascriptCode = `import { KMath } from '@kanaries/ml';

const loss = ([x, y]) => KMath.sum([x * x, 0.45 * y * y]);
const gradient = ([x, y]) => [2 * x, 0.9 * y];

let point = [-1.8, 1.7];
const learningRate = 0.08;

for (let step = 0; step < 60; step += 1) {
  const [dx, dy] = gradient(point);
  point = [
    point[0] - learningRate * dx,
    point[1] - learningRate * dy,
  ];
}`;

const pythonCode = `import numpy as np

def loss(point):
    x, y = point
    return np.sum([x * x, 0.45 * y * y])

def gradient(point):
    x, y = point
    return np.array([2 * x, 0.9 * y])

point = np.array([-1.8, 1.7])
learning_rate = 0.08

for step in range(60):
    point = point - learning_rate * gradient(point)`;

export function GradientDescentPlayground() {
  const [name, setName] = useState<ObjectiveName>('bowl');
  const [learningRate, setLearningRate] = useState(0.08);
  const [speed, setSpeed] = useState(160);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [start, setStart] = useState<Vector>([-1.8, 1.7]);
  const [dragging, setDragging] = useState(false);
  const paths = useMemo(() => trajectories(name, learningRate, start), [learningRate, name, start]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setStep((current) => {
      if (current >= 60) { setPlaying(false); return current; }
      return current + 1;
    }), speed);
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  const grid = useMemo(() => {
    const columns = 34;
    const rows = 28;
    const cells = Array.from({ length: columns * rows }, (_, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = -2.5 + (column + 0.5) * (5 / columns);
      const y = 2.5 - (row + 0.5) * (5 / rows);
      return { column, row, value: objective(name, [x, y]).value };
    });
    const rawMinimum = Math.min(...cells.map((cell) => cell.value));
    const values = cells.map((cell) => Math.log1p(Math.max(0, cell.value - rawMinimum)));
    return { cells, values, min: Math.min(...values), max: Math.max(...values), columns, rows };
  }, [name]);

  const toSvg = ([x, y]: Vector): [number, number] => [50 + (x + 2.5) / 5 * 500, 25 + (2.5 - y) / 5 * 350];
  const pointerPosition = (event: React.PointerEvent<SVGSVGElement>): Vector => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [Math.max(-2.5, Math.min(2.5, ((event.clientX - rect.left) / rect.width * 600 - 50) / 500 * 5 - 2.5)), Math.max(-2.5, Math.min(2.5, 2.5 - (((event.clientY - rect.top) / rect.height * 400 - 25) / 350 * 5)))];
  };
  const reset = () => { setStep(0); setPlaying(false); };

  return (
    <div className={styles.root} style={{ '--tool-accent': '#a94134' } as React.CSSProperties}>
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <div className={styles.controlGroup}><label htmlFor="gd-objective">Loss surface</label><select id="gd-objective" className={styles.select} value={name} onChange={(event) => { setName(event.target.value as ObjectiveName); reset(); }}><option value="bowl">Convex bowl</option><option value="rosenbrock">Rosenbrock valley</option><option value="rippled">Rippled non-convex</option></select></div>
          <div className={styles.controlGroup}><label htmlFor="gd-rate">Learning rate</label><div className={styles.rangeRow}><input id="gd-rate" type="range" min="0.01" max="0.18" step="0.01" value={learningRate} onChange={(event) => { setLearningRate(Number(event.target.value)); reset(); }} /><output>{learningRate.toFixed(2)}</output></div></div>
          <div className={styles.controlGroup}><label htmlFor="gd-speed">Animation speed</label><div className={styles.rangeRow}><input id="gd-speed" type="range" min="50" max="450" step="10" value={500 - speed} onChange={(event) => setSpeed(500 - Number(event.target.value))} /><output>{Math.round(1000 / speed)}×</output></div></div>
          <div className={styles.buttonRow}><button className={styles.buttonPrimary} type="button" onClick={() => { if (step >= 60) setStep(0); setPlaying((current) => !current); }}>{playing ? 'Pause' : step >= 60 ? 'Replay' : 'Play'}</button><button className={styles.button} type="button" onClick={() => { setPlaying(false); setStep((current) => Math.min(60, current + 1)); }}>Step</button><button className={styles.button} type="button" onClick={reset}>Reset</button></div>
          <p className={styles.hint}>Drag the white start marker, then play all three update rules from the same coordinates. Gradient norms are clipped at 8 to keep the comparison visible on steep surfaces.</p>
        </aside>

        <section className={styles.canvas}>
          <div className={styles.metricGrid}>
            <div className={styles.metric}><span>Iteration</span><strong>{step} / 60</strong></div>
            {OPTIMIZERS.slice(1).map((optimizer) => <div className={styles.metric} key={optimizer.key}><span>{optimizer.label} loss</span><strong>{paths[optimizer.key][step].value.toFixed(3)}</strong></div>)}
          </div>
          <h2 className={styles.chartTitle}>Optimizer trajectories on the loss surface</h2>
          <svg className={styles.svg} viewBox="0 0 600 400" role="img" aria-label="Animated comparison of SGD, Momentum, and Adam" onPointerMove={(event) => { if (dragging) { setStart(pointerPosition(event)); setStep(0); } }} onPointerUp={() => setDragging(false)} onPointerCancel={() => setDragging(false)}>
            {grid.cells.map((cell, index) => { const normalized = (grid.values[index] - grid.min) / Math.max(1e-9, grid.max - grid.min); return <rect key={index} x={50 + cell.column * 500 / grid.columns} y={25 + cell.row * 350 / grid.rows} width={500 / grid.columns + 0.5} height={350 / grid.rows + 0.5} fill={`hsl(${210 - normalized * 175} 55% ${94 - normalized * 38}%)`} />; })}
            <rect x="50" y="25" width="500" height="350" fill="none" stroke="currentColor" opacity="0.45" />
            {OPTIMIZERS.map((optimizer) => {
              const visible = paths[optimizer.key].slice(0, step + 1);
              const points = visible.map((point) => toSvg([point.x, point.y]).join(',')).join(' ');
              const current = toSvg([visible[visible.length - 1].x, visible[visible.length - 1].y]);
              return <g key={optimizer.key}><polyline points={points} fill="none" stroke="white" strokeWidth="5" opacity="0.72" /><polyline points={points} fill="none" stroke={optimizer.color} strokeWidth="2.5" /><circle cx={current[0]} cy={current[1]} r="6" fill={optimizer.color} stroke="white" strokeWidth="2" /></g>;
            })}
            {(() => { const [x, y] = toSvg(start); return <circle cx={x} cy={y} r="9" fill="none" stroke="white" strokeWidth="3" style={{ cursor: 'grab' }} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setPlaying(false); setDragging(true); }} />; })()}
            <text x="300" y="395" textAnchor="middle" className={styles.axisText}>parameter x</text><text x="12" y="200" textAnchor="middle" transform="rotate(-90 12 200)" className={styles.axisText}>parameter y</text>
          </svg>
          <div className={styles.legend}>{OPTIMIZERS.map((optimizer) => <span className={styles.legendItem} key={optimizer.key}><i className={styles.swatch} style={{ background: optimizer.color }} />{optimizer.label}: {paths[optimizer.key][step].value.toFixed(4)}</span>)}</div>
        </section>

        <section className={`${styles.card} ${styles.fullWidth}`}>
          <div className={styles.chartGrid}><div><h2 className={styles.chartTitle}>What the animation computes</h2><p className={styles.hint}>This is an explicit educational simulation of the published SGD, Momentum, and Adam update equations. <code>@kanaries/ml</code> supplies the browser-safe numerical sum used by each objective; the chart does not claim to expose private training trajectories from a library estimator.</p></div><div><h2 className={styles.chartTitle}>Same starting point, different state</h2><p className={styles.hint}>SGD uses only the current gradient. Momentum accumulates a velocity. Adam maintains bias-corrected first and second moments, adapting the step per coordinate. Their different state explains the diverging paths.</p></div></div>
        </section>
        <section className={`${styles.card} ${styles.fullWidth}`}><h2 className={styles.chartTitle}>Implement the update loop in JavaScript or Python</h2><CodeTabs javascript={javascriptCode} python={pythonCode} /></section>
      </div>
    </div>
  );
}
