import type { RefObject } from 'react';
import styles from './interactive.module.css';

type BoundaryChartProps = {
  X: number[][];
  y: number[];
  labels: number[];
  featureNames: string[];
  coefficients: number[];
  intercept: number;
  sample?: number[];
  svgRef?: RefObject<SVGSVGElement | null>;
};

const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const extent = (values: number[]) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.12, 0.5);
  return [min - padding, max + padding] as const;
};

export function BoundaryChart({
  X,
  y,
  labels,
  featureNames,
  coefficients,
  intercept,
  sample,
  svgRef,
}: BoundaryChartProps) {
  const width = 560;
  const height = 330;
  const margin = { top: 18, right: 18, bottom: 48, left: 54 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const [xMin, xMax] = extent(X.map((row) => row[0]));
  const sx = (value: number) => margin.left + ((value - xMin) / (xMax - xMin)) * innerWidth;

  if (featureNames.length === 1) {
    const sy = (value: number) => margin.top + (1 - value) * innerHeight;
    const curve = Array.from({ length: 81 }, (_, index) => {
      const x = xMin + (index / 80) * (xMax - xMin);
      return [sx(x), sy(sigmoid(intercept + coefficients[0] * x))];
    });
    const path = curve.map(([x, yy], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${yy.toFixed(2)}`).join(' ');
    return (
      <svg className={styles.chartSvg} ref={svgRef} role="img" aria-label="Fitted logistic sigmoid curve" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={width - margin.right} y1={sy(tick)} y2={sy(tick)} stroke="var(--color-fd-border)" />
            <text className={styles.axisLabel} x={margin.left - 9} y={sy(tick) + 3} textAnchor="end">{tick}</text>
          </g>
        ))}
        <line x1={margin.left} x2={width - margin.right} y1={sy(0.5)} y2={sy(0.5)} stroke="var(--color-fd-muted-foreground)" strokeDasharray="5 5" />
        <path d={path} fill="none" stroke="var(--tool-accent, #1d3f72)" strokeWidth="3" />
        {X.map((row, index) => {
          const positive = y[index] === labels[1];
          const jitter = ((index * 17) % 9 - 4) * 0.008;
          return (
            <circle
              key={index}
              cx={sx(row[0])}
              cy={sy((positive ? 0.94 : 0.06) + jitter)}
              r="5"
              fill={positive ? '#d45d4c' : '#2b7a78'}
              fillOpacity="0.82"
              stroke="var(--color-fd-card)"
              strokeWidth="1.5"
            >
              <title>{`${featureNames[0]} ${row[0]}, class ${y[index]}`}</title>
            </circle>
          );
        })}
        {sample && (
          <circle cx={sx(sample[0])} cy={sy(sigmoid(intercept + coefficients[0] * sample[0]))} r="7" fill="#f0a202" stroke="#5b3a00" strokeWidth="2">
            <title>Current sample prediction</title>
          </circle>
        )}
        <text className={styles.axisLabel} x={margin.left + innerWidth / 2} y={height - 12} textAnchor="middle">{featureNames[0]}</text>
        <text className={styles.axisLabel} x="14" y={margin.top + innerHeight / 2} textAnchor="middle" transform={`rotate(-90 14 ${margin.top + innerHeight / 2})`}>P(class {labels[1]})</text>
      </svg>
    );
  }

  const [yMin, yMax] = extent(X.map((row) => row[1]));
  const sy = (value: number) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * innerHeight;
  const gridSize = 20;
  const cells = Array.from({ length: gridSize * gridSize }, (_, index) => {
    const column = index % gridSize;
    const row = Math.floor(index / gridSize);
    const x0 = xMin + (column / gridSize) * (xMax - xMin);
    const x1 = xMin + ((column + 1) / gridSize) * (xMax - xMin);
    const y0 = yMin + (row / gridSize) * (yMax - yMin);
    const y1 = yMin + ((row + 1) / gridSize) * (yMax - yMin);
    const p = sigmoid(intercept + coefficients[0] * ((x0 + x1) / 2) + coefficients[1] * ((y0 + y1) / 2));
    return { x0, x1, y0, y1, p };
  });
  const boundary = [xMin, xMax].map((x) => ({
    x,
    y: Math.abs(coefficients[1]) < 1e-12 ? (yMin + yMax) / 2 : -(intercept + coefficients[0] * x) / coefficients[1],
  }));

  return (
    <svg className={styles.chartSvg} ref={svgRef} role="img" aria-label="Logistic regression decision boundary" viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <clipPath id="boundary-clip"><rect x={margin.left} y={margin.top} width={innerWidth} height={innerHeight} /></clipPath>
      </defs>
      <g clipPath="url(#boundary-clip)">
        {cells.map((cell, index) => (
          <rect
            key={index}
            x={sx(cell.x0)}
            y={sy(cell.y1)}
            width={Math.max(0, sx(cell.x1) - sx(cell.x0) + 0.5)}
            height={Math.max(0, sy(cell.y0) - sy(cell.y1) + 0.5)}
            fill={cell.p >= 0.5 ? '#d45d4c' : '#2b7a78'}
            fillOpacity={Number((0.06 + Math.abs(cell.p - 0.5) * 0.2).toFixed(6))}
          />
        ))}
        <line x1={sx(boundary[0].x)} y1={sy(boundary[0].y)} x2={sx(boundary[1].x)} y2={sy(boundary[1].y)} stroke="var(--tool-accent, #1d3f72)" strokeWidth="3" />
        {X.map((row, index) => (
          <circle
            key={index}
            cx={sx(row[0])}
            cy={sy(row[1])}
            r="5"
            fill={y[index] === labels[1] ? '#d45d4c' : '#2b7a78'}
            stroke="var(--color-fd-card)"
            strokeWidth="1.5"
          >
            <title>{`${featureNames[0]} ${row[0]}, ${featureNames[1]} ${row[1]}, class ${y[index]}`}</title>
          </circle>
        ))}
        {sample && sample.length >= 2 && (
          <path
            d={`M ${sx(sample[0])} ${sy(sample[1]) - 8} L ${sx(sample[0]) + 8} ${sy(sample[1])} L ${sx(sample[0])} ${sy(sample[1]) + 8} L ${sx(sample[0]) - 8} ${sy(sample[1])} Z`}
            fill="#f0a202"
            stroke="#5b3a00"
            strokeWidth="2"
          >
            <title>Current sample</title>
          </path>
        )}
      </g>
      <rect x={margin.left} y={margin.top} width={innerWidth} height={innerHeight} fill="none" stroke="var(--color-fd-border)" />
      <text className={styles.axisLabel} x={margin.left + innerWidth / 2} y={height - 12} textAnchor="middle">{featureNames[0]}</text>
      <text className={styles.axisLabel} x="14" y={margin.top + innerHeight / 2} textAnchor="middle" transform={`rotate(-90 14 ${margin.top + innerHeight / 2})`}>{featureNames[1]}</text>
    </svg>
  );
}
