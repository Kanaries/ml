'use client';

import { useMemo, useState } from 'react';
import { Neighbors } from '@kanaries/ml';
import { CodeTabs } from '@/components/tools/CodeTabs';
import { COLORS, makeBlobs, makeMoons, makeXor, type Point2D } from './data';
import styles from './playground.module.css';

type Metric = 'euclidean' | 'manhattan';
type Weight = 'uniform' | 'distance';
type Preset = 'blobs' | 'moons' | 'xor';

const presets: Record<Preset, () => Point2D[]> = {
  blobs: () => makeBlobs(54, [[-1.55, -1.15], [1.35, 1.15]], 0.58, 13),
  moons: () => makeMoons(64, 0.11, 23),
  xor: () => makeXor(60, 41),
};

const javascriptCode = `import { Neighbors } from '@kanaries/ml';

const classifier = new Neighbors.KNearestNeighbors(
  5,           // k
  'distance',  // 'uniform' or 'distance'
  'euclidean'
);

classifier.fit(trainX, trainY);
const labels = classifier.predict(testX);`;

const pythonCode = `from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)

classifier.fit(train_X, train_y)
labels = classifier.predict(test_X)`;

function distance(a: Point2D, b: Point2D, metric: Metric) {
  return metric === 'manhattan' ? Math.abs(a.x - b.x) + Math.abs(a.y - b.y) : Math.hypot(a.x - b.x, a.y - b.y);
}

export function KnnPlayground() {
  const [preset, setPreset] = useState<Preset>('moons');
  const [points, setPoints] = useState<Point2D[]>(() => presets.moons());
  const [query, setQuery] = useState<Point2D>({ x: 0.15, y: 0.25 });
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<Metric>('euclidean');
  const [weight, setWeight] = useState<Weight>('uniform');
  const [activeClass, setActiveClass] = useState(0);
  const [dragging, setDragging] = useState(false);

  const classifier = useMemo(() => {
    if (points.length === 0) return null;
    const model = new Neighbors.KNearestNeighbors(Math.min(k, points.length), weight, metric);
    model.fit(points.map((point) => [point.x, point.y]), points.map((point) => point.label ?? 0));
    return model;
  }, [k, metric, points, weight]);

  const fit = useMemo(() => {
    const columns = 36;
    const rows = 28;
    const cells = Array.from({ length: columns * rows }, (_, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      return { x: -3 + (column + 0.5) * (6 / columns), y: 3 - (row + 0.5) * (6 / rows), column, row };
    });
    const labels = classifier ? classifier.predict(cells.map((cell) => [cell.x, cell.y])) : cells.map(() => 0);
    return { cells, labels, columns, rows };
  }, [classifier]);
  const prediction = useMemo(() => classifier?.predict([[query.x, query.y]])[0] ?? 0, [classifier, query]);

  const neighbors = useMemo(() => points
    .map((point, index) => ({ point, index, distance: distance(point, query, metric) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.min(k, points.length)), [k, metric, points, query]);

  const toSvg = (point: Point2D): [number, number] => [50 + (point.x + 3) / 6 * 500, 25 + (3 - point.y) / 6 * 350];
  const fromClient = (clientX: number, clientY: number, svg: SVGSVGElement): Point2D => {
    const rect = svg.getBoundingClientRect();
    const sx = (clientX - rect.left) / rect.width * 600;
    const sy = (clientY - rect.top) / rect.height * 400;
    return { x: Math.max(-3, Math.min(3, (sx - 50) / 500 * 6 - 3)), y: Math.max(-3, Math.min(3, 3 - (sy - 25) / 350 * 6)) };
  };

  const loadPreset = (next: Preset) => {
    setPreset(next);
    setPoints(presets[next]());
    setQuery({ x: 0.15, y: 0.25 });
  };

  return (
    <div className={styles.root} style={{ '--tool-accent': '#624492' } as React.CSSProperties}>
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <div className={styles.controlGroup}>
            <span className={styles.label}>Dataset</span>
            <div className={styles.chips}>
              {(['blobs', 'moons', 'xor'] as Preset[]).map((name) => <button key={name} className={`${styles.chip} ${preset === name ? styles.chipActive : ''}`} type="button" onClick={() => loadPreset(name)}>{name}</button>)}
            </div>
          </div>
          <div className={styles.controlGroup}>
            <label htmlFor="knn-k">Neighbors (k)</label>
            <div className={styles.rangeRow}><input id="knn-k" type="range" min="1" max="15" step="2" value={k} onChange={(event) => setK(Number(event.target.value))} /><output>{k}</output></div>
          </div>
          <div className={styles.controlGroup}>
            <label htmlFor="knn-metric">Distance metric</label>
            <select id="knn-metric" className={styles.select} value={metric} onChange={(event) => setMetric(event.target.value as Metric)}><option value="euclidean">Euclidean (L2)</option><option value="manhattan">Manhattan (L1)</option></select>
          </div>
          <div className={styles.controlGroup}>
            <label htmlFor="knn-weight">Voting weights</label>
            <select id="knn-weight" className={styles.select} value={weight} onChange={(event) => setWeight(event.target.value as Weight)}><option value="uniform">Uniform vote</option><option value="distance">Inverse distance</option></select>
          </div>
          <div className={styles.controlGroup}>
            <span className={styles.label}>Click to add class</span>
            <div className={styles.buttonRow}>{[0, 1].map((label) => <button key={label} className={`${styles.chip} ${activeClass === label ? styles.chipActive : ''}`} type="button" onClick={() => setActiveClass(label)}><i className={styles.swatch} style={{ display: 'inline-block', marginRight: 5, background: COLORS[label] }} />Class {label}</button>)}</div>
          </div>
          <div className={styles.buttonRow}>
            <button className={styles.button} type="button" onClick={() => setPoints((current) => current.slice(0, -1))}>Undo point</button>
            <button className={styles.button} type="button" onClick={() => setPoints([])}>Clear</button>
          </div>
          <p className={styles.hint}>Click the decision map to add training samples. Drag the diamond query point to watch its nearest neighbors and prediction change.</p>
        </aside>

        <section className={styles.canvas}>
          <div className={styles.metricGrid}>
            <div className={styles.metric}><span>Prediction</span><strong style={{ color: COLORS[prediction] }}>{classifier ? `Class ${prediction}` : 'Add points'}</strong></div>
            <div className={styles.metric}><span>Training points</span><strong>{points.length}</strong></div>
            <div className={styles.metric}><span>Nearest distance</span><strong>{neighbors[0]?.distance.toFixed(3) ?? '—'}</strong></div>
          </div>
          <h2 className={styles.chartTitle}>KNN decision regions and local neighborhood</h2>
          <svg
            className={styles.svg}
            viewBox="0 0 600 400"
            role="img"
            aria-label="Interactive K nearest neighbors decision map"
            onPointerMove={(event) => { if (dragging) setQuery(fromClient(event.clientX, event.clientY, event.currentTarget)); }}
            onPointerUp={() => setDragging(false)}
            onPointerCancel={() => setDragging(false)}
          >
            {fit.cells.map((cell, index) => <rect key={index} x={50 + cell.column * (500 / fit.columns)} y={25 + cell.row * (350 / fit.rows)} width={500 / fit.columns + 0.5} height={350 / fit.rows + 0.5} fill={COLORS[fit.labels[index]]} opacity="0.13" />)}
            <rect
              x="50" y="25" width="500" height="350" fill="transparent" stroke="currentColor" opacity="0.4"
              onClick={(event) => { if (!dragging && event.currentTarget.ownerSVGElement) { const next = fromClient(event.clientX, event.clientY, event.currentTarget.ownerSVGElement); setPoints((current) => [...current, { ...next, label: activeClass }]); } }}
            />
            {neighbors.map(({ point, index }) => { const [x, y] = toSvg(point); const [qx, qy] = toSvg(query); return <line key={`line-${index}`} x1={qx} y1={qy} x2={x} y2={y} stroke={COLORS[point.label ?? 0]} strokeWidth="1.5" strokeDasharray="4 4" opacity="0.65" />; })}
            {points.map((point, index) => { const [x, y] = toSvg(point); return <circle key={index} cx={x} cy={y} r="5" fill={COLORS[point.label ?? 0]} stroke="white" strokeWidth="1.5" />; })}
            {(() => { const [x, y] = toSvg(query); return <rect x={x - 8} y={y - 8} width="16" height="16" rx="2" transform={`rotate(45 ${x} ${y})`} fill={COLORS[prediction]} stroke="white" strokeWidth="2.5" style={{ cursor: 'grab' }} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setDragging(true); }} />; })()}
            <text x="300" y="395" textAnchor="middle" className={styles.axisText}>Feature 1</text>
            <text x="12" y="200" textAnchor="middle" transform="rotate(-90 12 200)" className={styles.axisText}>Feature 2</text>
          </svg>
          <div className={styles.legend}>
            {[0, 1].map((label) => <span className={styles.legendItem} key={label}><i className={styles.swatch} style={{ background: COLORS[label] }} />Class {label}</span>)}
            <span className={styles.legendItem}>◆ draggable query</span>
            <span className={styles.legendItem}>dashed lines: {k} nearest</span>
          </div>
        </section>

        <section className={`${styles.card} ${styles.fullWidth}`}>
          <div className={styles.chartGrid}>
            <div>
              <h2 className={styles.chartTitle}>Current neighbor vote</h2>
              <div className={styles.tableWrap}><table className={styles.table}><thead><tr><th>Rank</th><th>Class</th><th>Distance</th><th>Vote weight</th></tr></thead><tbody>
                {neighbors.map((item, index) => <tr key={item.index}><td>{index + 1}</td><td style={{ color: COLORS[item.point.label ?? 0] }}>{item.point.label}</td><td>{item.distance.toFixed(3)}</td><td>{weight === 'uniform' ? '1.000' : (1 / Math.max(item.distance, 1e-12)).toFixed(3)}</td></tr>)}
              </tbody></table></div>
            </div>
            <div>
              <h2 className={styles.chartTitle}>What changes the boundary?</h2>
              <p className={styles.hint}><strong>k</strong> controls smoothness: small values follow local detail, while larger values average a wider neighborhood. <strong>Distance weighting</strong> gives close samples more influence. <strong>Manhattan distance</strong> creates diamond-shaped neighborhoods instead of Euclidean circles.</p>
              <p className={styles.hint}>The colored grid is predicted by a real <code>@kanaries/ml</code> <code>KNearestNeighbors</code> model. The table separately exposes the distances behind the vote.</p>
            </div>
          </div>
        </section>
        <section className={`${styles.card} ${styles.fullWidth}`}><h2 className={styles.chartTitle}>KNN in JavaScript and Python</h2><CodeTabs javascript={javascriptCode} python={pythonCode} /></section>
      </div>
    </div>
  );
}
