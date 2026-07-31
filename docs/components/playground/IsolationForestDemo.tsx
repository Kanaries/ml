'use client';

import { useMemo, useState } from 'react';
import { Ensemble } from '@kanaries/ml';
import { makeBlobs, type Point2D } from './data';
import styles from './playground.module.css';

const anomalies: Point2D[] = [{ x: -2.7, y: 2.5 }, { x: 2.7, y: -2.4 }, { x: 2.55, y: 2.65 }, { x: -2.6, y: -2.55 }, { x: 0.1, y: -2.75 }];

export function IsolationForestDemo() {
  const [trees, setTrees] = useState(60);
  const [contamination, setContamination] = useState(0.07);
  const [seed, setSeed] = useState(17);
  const points = useMemo(() => [...makeBlobs(72, [[0, 0]], 0.62, seed).map(({ x, y }) => ({ x, y })), ...anomalies], [seed]);
  const result = useMemo(() => {
    const X = points.map((point) => [point.x, point.y]);
    const model = new Ensemble.IsolationForest(Math.min(64, X.length), trees, contamination, 42);
    model.fit(X);
    const labels = model.predict(X);
    const scores = X.map((row) => model.anomalyScore(row));
    return { labels, scores, detected: labels.reduce<number>((sum, label) => sum + label, 0) };
  }, [contamination, points, trees]);
  const ranked = points.map((point, index) => ({ ...point, score: result.scores[index], anomaly: result.labels[index] })).sort((a, b) => b.score - a.score).slice(0, 6);

  return <div className={styles.root} style={{ '--tool-accent': '#a94134' } as React.CSSProperties}>
    <div className={styles.workspace}>
      <aside className={styles.controls}>
        <div className={styles.controlGroup}><label htmlFor="if-trees">Isolation trees</label><div className={styles.rangeRow}><input id="if-trees" type="range" min="10" max="120" step="10" value={trees} onChange={(event) => setTrees(Number(event.target.value))} /><output>{trees}</output></div></div>
        <div className={styles.controlGroup}><label htmlFor="if-contamination">Expected contamination</label><div className={styles.rangeRow}><input id="if-contamination" type="range" min="0.02" max="0.2" step="0.01" value={contamination} onChange={(event) => setContamination(Number(event.target.value))} /><output>{(contamination * 100).toFixed(0)}%</output></div></div>
        <button className={styles.button} type="button" onClick={() => setSeed((value) => value + 1)}>Generate new normal sample</button>
        <p className={styles.hint}>Five faraway points stay fixed while the central sample changes. Contamination sets the fitted score threshold, not the underlying score.</p>
      </aside>
      <section className={styles.canvas}>
        <div className={styles.metricGrid}><div className={styles.metric}><span>Observations</span><strong>{points.length}</strong></div><div className={styles.metric}><span>Flagged</span><strong>{result.detected}</strong></div><div className={styles.metric}><span>Top score</span><strong>{Math.max(...result.scores).toFixed(3)}</strong></div></div>
        <h2 className={styles.chartTitle}>Isolation Forest anomaly scores</h2>
        <svg className={styles.svg} viewBox="0 0 600 400" role="img" aria-label="Isolation Forest anomaly detection scatterplot">
          <rect x="50" y="25" width="500" height="350" fill="var(--color-fd-muted)" opacity="0.3" stroke="currentColor" />
          {points.map((point, index) => <g key={index}><circle cx={50 + (point.x + 3) / 6 * 500} cy={25 + (3 - point.y) / 6 * 350} r={5 + result.scores[index] * 6} fill={result.labels[index] ? '#a94134' : '#2b7a78'} stroke="white" strokeWidth="1.5" opacity="0.88" /><title>{`score ${result.scores[index].toFixed(4)} · ${result.labels[index] ? 'flagged anomaly' : 'normal'}`}</title></g>)}
          <text x="300" y="396" textAnchor="middle" className={styles.axisText}>Feature 1</text><text x="13" y="200" textAnchor="middle" transform="rotate(-90 13 200)" className={styles.axisText}>Feature 2</text>
        </svg>
        <div className={styles.legend}><span className={styles.legendItem}><i className={styles.swatch} style={{ background: '#2b7a78' }} />Normal</span><span className={styles.legendItem}><i className={styles.swatch} style={{ background: '#a94134' }} />Flagged anomaly</span><span className={styles.legendItem}>larger marker = higher score</span></div>
      </section>
      <section className={`${styles.card} ${styles.fullWidth}`}><h2 className={styles.chartTitle}>Highest anomaly scores</h2><div className={styles.tableWrap}><table className={styles.table}><thead><tr><th>Rank</th><th>x</th><th>y</th><th>Score</th><th>Threshold result</th></tr></thead><tbody>{ranked.map((point, index) => <tr key={`${point.x}-${point.y}`}><td>{index + 1}</td><td>{point.x.toFixed(3)}</td><td>{point.y.toFixed(3)}</td><td>{point.score.toFixed(4)}</td><td>{point.anomaly ? 'anomaly' : 'normal'}</td></tr>)}</tbody></table></div></section>
    </div>
  </div>;
}
