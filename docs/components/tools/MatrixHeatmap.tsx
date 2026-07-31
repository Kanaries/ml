import type { RefObject } from 'react';
import styles from './interactive.module.css';

type MatrixHeatmapProps = {
  matrix: number[][];
  labels: number[];
  svgRef?: RefObject<SVGSVGElement | null>;
};

export function MatrixHeatmap({ matrix, labels, svgRef }: MatrixHeatmapProps) {
  const size = Math.max(1, labels.length);
  const cell = Math.min(76, 260 / size);
  const margin = { top: 40, right: 12, bottom: 48, left: 56 };
  const width = margin.left + cell * size + margin.right;
  const height = margin.top + cell * size + margin.bottom;
  const max = Math.max(1, ...matrix.flat());

  return (
    <svg
      className={styles.chartSvg}
      ref={svgRef}
      role="img"
      aria-label={`Confusion matrix with ${size} classes. Rows are true labels and columns are predicted labels.`}
      viewBox={`0 0 ${width} ${height}`}
    >
      <text className={styles.axisLabel} x={margin.left + (cell * size) / 2} y={14} textAnchor="middle">
        Predicted label
      </text>
      <text
        className={styles.axisLabel}
        x={14}
        y={margin.top + (cell * size) / 2}
        textAnchor="middle"
        transform={`rotate(-90 14 ${margin.top + (cell * size) / 2})`}
      >
        True label
      </text>
      {labels.map((label, index) => (
        <g key={`label-${label}`}>
          <text className={styles.axisLabel} x={margin.left + index * cell + cell / 2} y={margin.top - 10} textAnchor="middle">
            {label}
          </text>
          <text className={styles.axisLabel} x={margin.left - 10} y={margin.top + index * cell + cell / 2 + 3} textAnchor="end">
            {label}
          </text>
        </g>
      ))}
      {matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
        const intensity = 0.12 + (value / max) * 0.78;
        const textColor = intensity > 0.55 ? '#fff' : 'currentColor';
        return (
          <g key={`${rowIndex}-${columnIndex}`}>
            <rect
              x={margin.left + columnIndex * cell}
              y={margin.top + rowIndex * cell}
              width={cell - 2}
              height={cell - 2}
              rx={4}
              fill={`color-mix(in srgb, var(--tool-accent, #1d3f72) ${Math.round(intensity * 100)}%, var(--color-fd-card))`}
            >
              <title>{`True ${labels[rowIndex]}, predicted ${labels[columnIndex]}: ${value}`}</title>
            </rect>
            <text
              x={margin.left + columnIndex * cell + cell / 2 - 1}
              y={margin.top + rowIndex * cell + cell / 2 + 4}
              textAnchor="middle"
              fill={textColor}
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fontSize={Math.max(9, Math.min(14, cell / 4))}
              pointerEvents="none"
            >
              {value}
            </text>
          </g>
        );
      }))}
    </svg>
  );
}
