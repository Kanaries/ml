'use client';

import { useMemo, useState } from 'react';
import { Decomposition } from '@kanaries/ml';
import { CodeTabs } from '@/components/tools/CodeTabs';
import { DataInput, parseNumericTable } from '@/components/tools/DataInput';
import styles from './playground.module.css';

type PcaResult =
  | { ok: true; parsed: ReturnType<typeof parseNumericTable>; model: Decomposition.PCA; scores: number[][]; components: number[][]; explained: number[]; totalVariance: number }
  | { ok: false; error: string };

const flowerData = `sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
5.4,3.9,1.7,0.4
5.0,3.4,1.5,0.2
7.0,3.2,4.7,1.4
6.4,3.2,4.5,1.5
6.9,3.1,4.9,1.5
5.5,2.3,4.0,1.3
6.5,3.0,5.8,2.2
7.6,3.0,6.6,2.1
4.9,2.5,4.5,1.7
7.3,2.9,6.3,1.8`;

const productData = `sessions,minutes,actions,revenue
3,12,8,0
5,19,14,12
7,31,22,18
9,38,29,35
11,44,37,41
14,58,46,59
16,62,51,63
18,75,63,82
21,81,70,91
24,96,79,118`;

function rotate([x, y]: number[], angle: number): [number, number] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

function format(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : '—';
}

const javascriptCode = `import { Decomposition } from '@kanaries/ml';

const pca = new Decomposition.PCA(2);
const projected = pca.fitTransform(X);

console.log(projected);
console.log(pca.getComponents());
console.log(pca.getExplainedVariance());

// Map a 2D point back into the original feature space.
const reconstructed = pca.inverseTransform([projected[0]]);`;

const pythonCode = `from sklearn.decomposition import PCA

pca = PCA(n_components=2)
projected = pca.fit_transform(X)

print(projected)
print(pca.components_)
print(pca.explained_variance_)

# Map a 2D point back into the original feature space.
reconstructed = pca.inverse_transform(projected[0:1])`;

export function PcaPlayground() {
  const [input, setInput] = useState(flowerData);
  const [angle, setAngle] = useState(0);
  const [dragFeature, setDragFeature] = useState<number | null>(null);
  const [selected, setSelected] = useState(0);

  const result = useMemo<PcaResult>(() => {
    try {
      const parsed = parseNumericTable(input);
      if (parsed.rows.length < 3) throw new Error('PCA needs at least three rows.');
      if (parsed.headers.length < 2) throw new Error('PCA needs at least two numeric columns.');
      const model = new Decomposition.PCA(2);
      const scores = model.fitTransform(parsed.rows);
      const components = model.getComponents();
      const explained = model.getExplainedVariance();
      const mean = model.getMean();
      const totalVariance = parsed.headers.reduce((total, _, column) => {
        const sumSquares = parsed.rows.reduce((sum, row) => sum + (row[column] - mean[column]) ** 2, 0);
        return total + sumSquares / Math.max(1, parsed.rows.length - 1);
      }, 0);
      return { ok: true, parsed, model, scores, components, explained, totalVariance };
    } catch (caught) {
      return { ok: false, error: caught instanceof Error ? caught.message : 'PCA could not fit this dataset.' };
    }
  }, [input]);

  if (!result.ok) {
    return <div className={styles.root}><div className={styles.error}>{result.error}</div></div>;
  }

  const { parsed, model, scores, components, explained, totalVariance } = result;
  const rotatedScores = scores.map((point) => rotate(point, angle));
  const maxAbs = Math.max(1, ...rotatedScores.flatMap(([x, y]) => [Math.abs(x), Math.abs(y)]));
  const plotScale = 150 / maxAbs;
  const cx = 300;
  const cy = 195;
  const safeSelected = Math.min(selected, scores.length - 1);
  const reconstructed = model.inverseTransform([scores[safeSelected]])[0];
  const reconstructionError = Math.sqrt(parsed.rows[safeSelected].reduce((sum, value, index) => sum + (value - reconstructed[index]) ** 2, 0));
  const loadingScale = 115;

  const updateDrag = (event: React.PointerEvent<SVGSVGElement>) => {
    if (dragFeature === null) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 600 - cx;
    const y = -(((event.clientY - rect.top) / rect.height) * 390 - cy);
    const base = Math.atan2(components[1][dragFeature], components[0][dragFeature]);
    setAngle(Math.atan2(y, x) - base);
  };

  return (
    <div className={styles.root} style={{ '--tool-accent': '#17605e' } as React.CSSProperties}>
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <DataInput
            value={input}
            onChange={(value) => { setInput(value); setSelected(0); setAngle(0); }}
            label="Multidimensional dataset"
            hint="Paste comma-, tab-, or semicolon-separated numeric data. Columns are features and rows are observations."
            examples={[{ label: 'flowers', value: flowerData }, { label: 'product metrics', value: productData }]}
            maxPreviewRows={5}
          />
          <div className={styles.controlGroup}>
            <span className={styles.label}>Projection controls</span>
            <button className={styles.button} type="button" onClick={() => setAngle(0)}>Reset loading rotation</button>
            <p className={styles.hint}>Drag any loading arrow. The complete basis rotates with it, so distances, angles, and the fitted PCA solution remain valid.</p>
          </div>
        </aside>

        <section className={styles.canvas}>
          <div className={styles.metricGrid}>
            <div className={styles.metric}><span>Rows × features</span><strong>{parsed.rows.length} × {parsed.headers.length}</strong></div>
            <div className={styles.metric}><span>PC1 variance</span><strong>{format(explained[0] / totalVariance * 100)}%</strong></div>
            <div className={styles.metric}><span>PC1 + PC2</span><strong>{format((explained[0] + explained[1]) / totalVariance * 100)}%</strong></div>
          </div>
          <h2 className={styles.chartTitle}>PCA biplot — select a point or drag a loading vector</h2>
          <svg
            className={styles.svg}
            viewBox="0 0 600 390"
            role="img"
            aria-label="Two-dimensional PCA projection with draggable feature loading vectors"
            onPointerMove={updateDrag}
            onPointerUp={() => setDragFeature(null)}
            onPointerCancel={() => setDragFeature(null)}
            onPointerLeave={() => setDragFeature(null)}
          >
            <line x1="35" y1={cy} x2="575" y2={cy} stroke="currentColor" opacity="0.18" />
            <line x1={cx} y1="20" x2={cx} y2="365" stroke="currentColor" opacity="0.18" />
            <text x="565" y={cy - 8} className={styles.axisText}>PC1</text>
            <text x={cx + 8} y="30" className={styles.axisText}>PC2</text>
            {rotatedScores.map(([x, y], index) => (
              <circle
                key={index}
                cx={cx + x * plotScale}
                cy={cy - y * plotScale}
                r={safeSelected === index ? 7 : 4.5}
                fill={safeSelected === index ? '#d45d4c' : '#2b7a78'}
                stroke="white"
                strokeWidth={safeSelected === index ? 2 : 0.8}
                opacity={safeSelected === index ? 1 : 0.78}
                tabIndex={0}
                role="button"
                aria-label={`Select observation ${index + 1}`}
                onClick={() => setSelected(index)}
                onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') setSelected(index); }}
              />
            ))}
            {parsed.headers.map((header, feature) => {
              const [x, y] = rotate([components[0][feature], components[1][feature]], angle);
              const endX = cx + x * loadingScale;
              const endY = cy - y * loadingScale;
              return (
                <g key={header}>
                  <line x1={cx} y1={cy} x2={endX} y2={endY} stroke="#d45d4c" strokeWidth="2.4" />
                  <circle
                    cx={endX}
                    cy={endY}
                    r="8"
                    fill="#d45d4c"
                    stroke="white"
                    strokeWidth="2"
                    style={{ cursor: 'grab' }}
                    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setDragFeature(feature); }}
                  />
                  <text x={endX + (x >= 0 ? 10 : -10)} y={endY - 8} textAnchor={x >= 0 ? 'start' : 'end'} className={styles.axisText}>{header}</text>
                </g>
              );
            })}
          </svg>
          <div className={styles.legend}>
            <span className={styles.legendItem}><i className={styles.swatch} style={{ background: '#2b7a78' }} />Projected observations</span>
            <span className={styles.legendItem}><i className={styles.swatch} style={{ background: '#d45d4c' }} />Selected observation / feature loadings</span>
          </div>
        </section>

        <section className={`${styles.card} ${styles.fullWidth}`}>
          <div className={styles.chartGrid}>
            <div>
              <h2 className={styles.chartTitle}>Bidirectional projection: observation {safeSelected + 1}</h2>
              <div className={styles.tableWrap}>
                <table className={styles.table}><thead><tr><th>Feature</th><th>Original</th><th>2D reconstruction</th></tr></thead><tbody>
                  {parsed.headers.map((header, index) => <tr key={header}><td>{header}</td><td>{format(parsed.rows[safeSelected][index])}</td><td>{format(reconstructed[index])}</td></tr>)}
                </tbody></table>
              </div>
              <p className={styles.hint}>Euclidean reconstruction error: <strong>{format(reconstructionError)}</strong>. A larger value means the omitted components carried more information for this row.</p>
            </div>
            <div>
              <h2 className={styles.chartTitle}>Explained variance by retained component</h2>
              {[0, 1].map((index) => (
                <div className={styles.controlGroup} key={index}>
                  <span className={styles.label}>PC{index + 1} · {format(explained[index] / totalVariance * 100)}%</span>
                  <div style={{ height: 12, borderRadius: 6, background: 'var(--color-fd-muted)', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, explained[index] / totalVariance * 100)}%`, height: '100%', background: index === 0 ? '#2b7a78' : '#d45d4c' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
        <section className={`${styles.card} ${styles.fullWidth}`}>
          <h2 className={styles.chartTitle}>Use the same PCA in JavaScript or Python</h2>
          <CodeTabs javascript={javascriptCode} python={pythonCode} />
        </section>
      </div>
    </div>
  );
}
