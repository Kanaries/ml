'use client';

import { useEffect, useMemo, useState } from 'react';
import { Clusters } from '@kanaries/ml';
import { CodeTabs } from '@/components/tools/CodeTabs';
import { COLORS, makeBlobs, makeMoons, type Point2D } from './data';
import styles from './playground.module.css';

type Preset = 'blobs' | 'moons' | 'uneven';
type Snapshot = { centers: Point2D[]; labels: number[]; inertia: number };

const presets: Record<Preset, () => Point2D[]> = {
  blobs: () => makeBlobs(72, [[-1.75, -1.25], [1.65, -0.7], [0.15, 1.65]], 0.42, 11).map(({ x, y }) => ({ x, y })),
  moons: () => makeMoons(76, 0.09, 23).map(({ x, y }) => ({ x: x * 1.35 - 0.65, y: y * 1.55 })),
  uneven: () => [
    ...makeBlobs(52, [[-1.55, -0.9]], 0.58, 51),
    ...makeBlobs(18, [[1.65, -0.4]], 0.24, 61),
    ...makeBlobs(10, [[0.55, 1.8]], 0.18, 71),
  ].map(({ x, y }) => ({ x, y })),
};

function squaredDistance(a: Point2D, b: Point2D) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function initialCenters(points: Point2D[], k: number) {
  const centers = [{ ...points[0] }];
  while (centers.length < k) {
    const next = points.reduce((best, point) => {
      const distance = Math.min(...centers.map((center) => squaredDistance(point, center)));
      return distance > best.distance ? { point, distance } : best;
    }, { point: points[0], distance: -1 });
    centers.push({ ...next.point });
  }
  return centers;
}

function lloydHistory(points: Point2D[], k: number): { snapshots: Snapshot[]; initial: Point2D[] } {
  const initial = initialCenters(points, k);
  let centers = initial.map((center) => ({ ...center }));
  const snapshots: Snapshot[] = [];
  for (let iteration = 0; iteration < 30; iteration += 1) {
    const labels = points.map((point) => centers.reduce((best, center, index) => {
      const distance = squaredDistance(point, center);
      return distance < best.distance ? { index, distance } : best;
    }, { index: 0, distance: Number.POSITIVE_INFINITY }).index);
    const inertia = points.reduce((sum, point, index) => sum + squaredDistance(point, centers[labels[index]]), 0);
    snapshots.push({ centers: centers.map((center) => ({ ...center })), labels, inertia });
    const next = centers.map((center, cluster) => {
      const members = points.filter((_, index) => labels[index] === cluster);
      if (members.length === 0) return center;
      return { x: members.reduce((sum, point) => sum + point.x, 0) / members.length, y: members.reduce((sum, point) => sum + point.y, 0) / members.length };
    });
    const shift = Math.max(...centers.map((center, index) => Math.sqrt(squaredDistance(center, next[index]))));
    centers = next;
    if (shift < 1e-4) {
      const finalLabels = points.map((point) => centers.reduce((best, center, index) => {
        const distance = squaredDistance(point, center);
        return distance < best.distance ? { index, distance } : best;
      }, { index: 0, distance: Number.POSITIVE_INFINITY }).index);
      snapshots.push({ centers: centers.map((center) => ({ ...center })), labels: finalLabels, inertia: points.reduce((sum, point, index) => sum + squaredDistance(point, centers[finalLabels[index]]), 0) });
      break;
    }
  }
  return { snapshots, initial };
}

const javascriptCode = `import { Clusters } from '@kanaries/ml';

const model = new Clusters.KMeans(
  3,       // number of clusters
  1e-4,    // convergence tolerance
  undefined,
  100,     // maximum iterations
  42,      // random seed
  10       // k-means++ restarts
);

const labels = model.fitPredict(X);
console.log(model.getCentroids());
console.log(model.getInertia());`;

const pythonCode = `from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    tol=1e-4,
    max_iter=100,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X)
print(model.cluster_centers_)
print(model.inertia_)`;

export function KMeansPlayground() {
  const [preset, setPreset] = useState<Preset>('blobs');
  const [points, setPoints] = useState<Point2D[]>(() => presets.blobs());
  const [k, setK] = useState(3);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(550);
  const history = useMemo(() => lloydHistory(points, k), [k, points]);
  const safeStep = Math.min(step, history.snapshots.length - 1);
  const snapshot = history.snapshots[safeStep];

  const validation = useMemo(() => {
    const model = new Clusters.KMeans(k, 1e-4, history.initial.map((point) => [point.x, point.y]), 30);
    const labels = model.fitPredict(points.map((point) => [point.x, point.y]));
    const centroids = model.getCentroids() ?? [];
    const centerDelta = Math.max(...centroids.map((center, index) => Math.hypot(center[0] - history.snapshots.at(-1)!.centers[index].x, center[1] - history.snapshots.at(-1)!.centers[index].y)));
    return { labels, inertia: model.getInertia(), centerDelta };
  }, [history, k, points]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setStep((current) => {
      if (current >= history.snapshots.length - 1) { setPlaying(false); return current; }
      return current + 1;
    }), speed);
    return () => window.clearInterval(timer);
  }, [history.snapshots.length, playing, speed]);

  const toSvg = (point: Point2D): [number, number] => [50 + (point.x + 3) / 6 * 500, 25 + (3 - point.y) / 6 * 350];
  const reset = () => { setStep(0); setPlaying(false); };
  const loadPreset = (next: Preset) => { setPreset(next); setPoints(presets[next]()); reset(); };

  return (
    <div className={styles.root} style={{ '--tool-accent': '#236a9d' } as React.CSSProperties}>
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <div className={styles.controlGroup}><span className={styles.label}>Dataset</span><div className={styles.chips}>{(['blobs', 'moons', 'uneven'] as Preset[]).map((name) => <button key={name} className={`${styles.chip} ${preset === name ? styles.chipActive : ''}`} type="button" onClick={() => loadPreset(name)}>{name}</button>)}</div></div>
          <div className={styles.controlGroup}><label htmlFor="km-k">Number of clusters (k)</label><div className={styles.rangeRow}><input id="km-k" type="range" min="2" max="5" value={k} onChange={(event) => { setK(Number(event.target.value)); reset(); }} /><output>{k}</output></div></div>
          <div className={styles.controlGroup}><label htmlFor="km-speed">Step duration</label><div className={styles.rangeRow}><input id="km-speed" type="range" min="150" max="950" step="50" value={1100 - speed} onChange={(event) => setSpeed(1100 - Number(event.target.value))} /><output>{speed}ms</output></div></div>
          <div className={styles.buttonRow}><button className={styles.buttonPrimary} type="button" onClick={() => { if (safeStep >= history.snapshots.length - 1) setStep(0); setPlaying((current) => !current); }}>{playing ? 'Pause' : 'Play'}</button><button className={styles.button} type="button" onClick={() => { setPlaying(false); setStep((current) => Math.min(history.snapshots.length - 1, current + 1)); }}>Next iteration</button><button className={styles.button} type="button" onClick={reset}>Reset</button></div>
          <p className={styles.hint}>Click the plot to add observations. Reset shows deterministic farthest-first seeds; each step alternates assignment and centroid recomputation as one Lloyd iteration.</p>
        </aside>

        <section className={styles.canvas}>
          <div className={styles.metricGrid}>
            <div className={styles.metric}><span>Iteration</span><strong>{safeStep} / {history.snapshots.length - 1}</strong></div>
            <div className={styles.metric}><span>Inertia (SSE)</span><strong>{snapshot.inertia.toFixed(2)}</strong></div>
            <div className={styles.metric}><span>Library delta</span><strong>{validation.centerDelta.toExponential(1)}</strong></div>
          </div>
          <h2 className={styles.chartTitle}>Lloyd iterations: assignments and moving centroids</h2>
          <svg className={styles.svg} viewBox="0 0 600 400" role="img" aria-label="Interactive K-Means clustering iterations">
            <rect x="50" y="25" width="500" height="350" rx="4" fill="var(--color-fd-muted)" opacity="0.35" stroke="currentColor" onClick={(event) => { const svg = event.currentTarget.ownerSVGElement; if (!svg) return; const rect = svg.getBoundingClientRect(); const sx = (event.clientX - rect.left) / rect.width * 600; const sy = (event.clientY - rect.top) / rect.height * 400; setPoints((current) => [...current, { x: Math.max(-3, Math.min(3, (sx - 50) / 500 * 6 - 3)), y: Math.max(-3, Math.min(3, 3 - (sy - 25) / 350 * 6)) }]); reset(); }} />
            {points.map((point, index) => { const [x, y] = toSvg(point); return <circle key={index} cx={x} cy={y} r="5" fill={COLORS[snapshot.labels[index]]} stroke="white" strokeWidth="1.2" opacity="0.84" />; })}
            {snapshot.centers.map((center, cluster) => { const [x, y] = toSvg(center); return <g key={cluster}><circle cx={x} cy={y} r="13" fill={COLORS[cluster]} stroke="white" strokeWidth="3" /><path d={`M ${x - 6} ${y} L ${x + 6} ${y} M ${x} ${y - 6} L ${x} ${y + 6}`} stroke="white" strokeWidth="2.5" /></g>; })}
            <text x="300" y="395" textAnchor="middle" className={styles.axisText}>Feature 1</text><text x="12" y="200" textAnchor="middle" transform="rotate(-90 12 200)" className={styles.axisText}>Feature 2</text>
          </svg>
          <div className={styles.legend}>{snapshot.centers.map((_, index) => <span className={styles.legendItem} key={index}><i className={styles.swatch} style={{ background: COLORS[index] }} />Cluster {index + 1}</span>)}</div>
        </section>

        <section className={`${styles.card} ${styles.fullWidth}`}>
          <div className={styles.chartGrid}>
            <div><h2 className={styles.chartTitle}>Inertia across iterations</h2><svg className={styles.svg} viewBox="0 0 500 170" role="img" aria-label="K-Means inertia history"><line x1="35" y1="140" x2="480" y2="140" stroke="currentColor" opacity="0.3" /><polyline fill="none" stroke="#3480b8" strokeWidth="3" points={history.snapshots.slice(0, safeStep + 1).map((item, index) => { const max = Math.max(...history.snapshots.map((entry) => entry.inertia)); const min = Math.min(...history.snapshots.map((entry) => entry.inertia)); return `${35 + index / Math.max(1, history.snapshots.length - 1) * 445},${140 - (item.inertia - min) / Math.max(1e-9, max - min) * 125}`; }).join(' ')} /></svg></div>
            <div><h2 className={styles.chartTitle}>Implementation check</h2><p className={styles.hint}>The animation records transparent Lloyd assignment/update steps. A real <code>@kanaries/ml</code> <code>KMeans</code> model is also fit with the identical initial centers. Final centroid difference: <strong>{validation.centerDelta.toExponential(3)}</strong>; library inertia: <strong>{validation.inertia.toFixed(3)}</strong>.</p><p className={styles.hint}>The moons preset demonstrates a limitation: centroid-based Voronoi regions cannot follow arbitrary curved clusters.</p></div>
          </div>
        </section>
        <section className={`${styles.card} ${styles.fullWidth}`}><h2 className={styles.chartTitle}>Fit K-Means in JavaScript or Python</h2><CodeTabs javascript={javascriptCode} python={pythonCode} /></section>
      </div>
    </div>
  );
}
