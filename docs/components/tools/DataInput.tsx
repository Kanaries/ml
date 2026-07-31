'use client';

import styles from './interactive.module.css';

export type ParsedTable = {
  headers: string[];
  rows: number[][];
};

export function parseNumericTable(value: string): ParsedTable {
  const lines = value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    throw new Error('Paste at least one row of numeric data.');
  }

  const split = (line: string) => line.split(/\s*[,;\t]\s*|\s{2,}/).map((cell) => cell.trim());
  const first = split(lines[0]);
  const hasHeader = first.some((cell) => cell === '' || !Number.isFinite(Number(cell)));
  const headers = hasHeader ? first : first.map((_, index) => `column_${index + 1}`);
  const dataLines = hasHeader ? lines.slice(1) : lines;
  const rows = dataLines.map((line, rowIndex) => {
    const cells = split(line);
    if (cells.length !== headers.length) {
      throw new Error(`Row ${rowIndex + 1} has ${cells.length} columns; expected ${headers.length}.`);
    }
    const row = cells.map(Number);
    if (row.some((cell) => !Number.isFinite(cell))) {
      throw new Error(`Row ${rowIndex + 1} contains a non-numeric value.`);
    }
    return row;
  });

  if (rows.length === 0) {
    throw new Error('Add at least one numeric data row below the header.');
  }

  return { headers, rows };
}

type DataInputProps = {
  value: string;
  onChange: (value: string) => void;
  label: string;
  hint: string;
  examples?: Array<{ label: string; value: string }>;
  maxPreviewRows?: number;
};

export function DataInput({
  value,
  onChange,
  label,
  hint,
  examples = [],
  maxPreviewRows = 8,
}: DataInputProps) {
  let parsed: ParsedTable | null = null;
  let error = '';
  try {
    parsed = parseNumericTable(value);
  } catch (caught) {
    error = caught instanceof Error ? caught.message : 'Could not parse this data.';
  }

  const editCell = (rowIndex: number, columnIndex: number, nextValue: string) => {
    if (!parsed) return;
    const nextRows = parsed.rows.map((row) => row.slice());
    nextRows[rowIndex][columnIndex] = Number(nextValue);
    onChange([parsed.headers.join(','), ...nextRows.map((row) => row.join(','))].join('\n'));
  };

  return (
    <div>
      <div className={styles.inputLabel}>{label}</div>
      <div className={styles.hint}>{hint}</div>
      {examples.length > 0 && (
        <div className={styles.sampleActions} aria-label="Example datasets">
          {examples.map((example) => (
            <button className={styles.sampleButton} key={example.label} onClick={() => onChange(example.value)} type="button">
              Load {example.label}
            </button>
          ))}
        </div>
      )}
      <textarea
        className={styles.textarea}
        aria-label={label}
        spellCheck={false}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {error && <div className={styles.error}>{error}</div>}
      {parsed && (
        <div className={styles.preview}>
          <table className={styles.table}>
            <thead>
              <tr>
                {parsed.headers.map((header) => <th key={header}>{header}</th>)}
              </tr>
            </thead>
            <tbody>
              {parsed.rows.slice(0, maxPreviewRows).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, columnIndex) => (
                    <td key={columnIndex}>
                      <input
                        className={styles.tableInput}
                        type="number"
                        step="any"
                        value={cell}
                        aria-label={`Row ${rowIndex + 1}, ${parsed?.headers[columnIndex]}`}
                        onChange={(event) => editCell(rowIndex, columnIndex, event.target.value)}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
