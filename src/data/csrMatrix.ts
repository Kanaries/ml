import { registerSerializableClass } from '../base/estimator';

/** A dense numeric feature matrix, indexed as [sample][feature]. */
export type DenseMatrix = number[][];

/** Numeric input accepted by sparse-aware estimators. */
export type NumericMatrix = DenseMatrix | CSRMatrix;

/** Raw text documents accepted by text transformers. */
export type TextDocuments = string[];

/** Data that may flow between Pipeline steps. */
export type FeatureData = NumericMatrix | TextDocuments;
export type FeatureDataKind = 'dense' | 'csr' | 'text';

/**
 * Minimal compressed-sparse-row matrix for browser and Node.js workloads.
 *
 * The three backing arrays use the conventional CSR layout: values and their
 * column indices are grouped by row, while indptr stores the start offset for
 * every row (plus one final sentinel). Zero values are not stored.
 */
export class CSRMatrix {
    private dataState: number[];
    private indicesState: number[];
    private indptrState: number[];
    public readonly nRows: number;
    public readonly nCols: number;

    constructor(
        data: number[],
        indices: number[],
        indptr: number[],
        shape: [number, number],
    ) {
        const [nRows, nCols] = shape;
        if (!Number.isInteger(nRows) || nRows < 0 || !Number.isInteger(nCols) || nCols < 0) {
            throw new Error('CSR shape must contain non-negative integer dimensions');
        }
        if (data.length !== indices.length) {
            throw new Error('CSR data and indices must have the same length');
        }
        if (indptr.length !== nRows + 1 || indptr[0] !== 0 || indptr[nRows] !== data.length) {
            throw new Error('CSR indptr must have nRows + 1 entries spanning all stored values');
        }
        for (let row = 0; row < nRows; row++) {
            if (!Number.isInteger(indptr[row]) || indptr[row] > indptr[row + 1]) {
                throw new Error('CSR indptr must be a non-decreasing integer sequence');
            }
            let previous = -1;
            for (let p = indptr[row]; p < indptr[row + 1]; p++) {
                const column = indices[p];
                if (!Number.isInteger(column) || column < 0 || column >= nCols) {
                    throw new Error(`CSR column index ${column} is out of bounds for ${nCols} columns`);
                }
                if (column <= previous) {
                    throw new Error('CSR column indices must be strictly increasing within each row');
                }
                if (data[p] === 0) {
                    throw new Error('CSR data must not store explicit zero values');
                }
                previous = column;
            }
        }
        this.dataState = data.slice();
        this.indicesState = indices.slice();
        this.indptrState = indptr.slice();
        this.nRows = nRows;
        this.nCols = nCols;
    }

    public get shape(): [number, number] {
        return [this.nRows, this.nCols];
    }

    public get nnz(): number {
        return this.dataState.length;
    }

    /** Defensive copies keep callers from invalidating the CSR layout. */
    public get data(): readonly number[] {
        return this.dataState.slice();
    }

    public get indices(): readonly number[] {
        return this.indicesState.slice();
    }

    public get indptr(): readonly number[] {
        return this.indptrState.slice();
    }

    public static fromDense(matrix: DenseMatrix): CSRMatrix {
        const nRows = matrix.length;
        const nCols = nRows === 0 ? 0 : matrix[0].length;
        const data: number[] = [];
        const indices: number[] = [];
        const indptr: number[] = [0];
        for (const row of matrix) {
            if (row.length !== nCols) {
                throw new Error('all rows in a dense matrix must have the same length');
            }
            for (let column = 0; column < nCols; column++) {
                if (row[column] !== 0) {
                    data.push(row[column]);
                    indices.push(column);
                }
            }
            indptr.push(data.length);
        }
        return new CSRMatrix(data, indices, indptr, [nRows, nCols]);
    }

    public get(row: number, column: number): number {
        this.assertRow(row);
        if (!Number.isInteger(column) || column < 0 || column >= this.nCols) {
            throw new Error(`column index ${column} is out of bounds`);
        }
        let lo = this.indptrState[row];
        let hi = this.indptrState[row + 1] - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >>> 1;
            const current = this.indicesState[mid];
            if (current === column) return this.dataState[mid];
            if (current < column) lo = mid + 1;
            else hi = mid - 1;
        }
        return 0;
    }

    public row(row: number): number[] {
        this.assertRow(row);
        const dense = new Array(this.nCols).fill(0);
        this.forEachNonZeroInRow(row, (column, value) => {
            dense[column] = value;
        });
        return dense;
    }

    public forEachNonZeroInRow(row: number, callback: (column: number, value: number) => void): void {
        this.assertRow(row);
        for (let p = this.indptrState[row]; p < this.indptrState[row + 1]; p++) {
            callback(this.indicesState[p], this.dataState[p]);
        }
    }

    public selectRows(rows: number[]): CSRMatrix {
        const data: number[] = [];
        const indices: number[] = [];
        const indptr: number[] = [0];
        for (const row of rows) {
            this.assertRow(row);
            for (let p = this.indptrState[row]; p < this.indptrState[row + 1]; p++) {
                data.push(this.dataState[p]);
                indices.push(this.indicesState[p]);
            }
            indptr.push(data.length);
        }
        return new CSRMatrix(data, indices, indptr, [rows.length, this.nCols]);
    }

    public toDense(): DenseMatrix {
        return Array.from({ length: this.nRows }, (_, row) => this.row(row));
    }

    private assertRow(row: number): void {
        if (!Number.isInteger(row) || row < 0 || row >= this.nRows) {
            throw new Error(`row index ${row} is out of bounds`);
        }
    }
}

export function isCSRMatrix(value: unknown): value is CSRMatrix {
    return value instanceof CSRMatrix;
}

export function matrixShape(X: NumericMatrix): [number, number] {
    if (isCSRMatrix(X)) return X.shape;
    return [X.length, X.length === 0 ? 0 : X[0].length];
}

export function matrixRow(X: NumericMatrix, row: number): number[] {
    return isCSRMatrix(X) ? X.row(row) : X[row];
}

export function forEachNonZeroInRow(
    X: NumericMatrix,
    row: number,
    callback: (column: number, value: number) => void,
): void {
    if (isCSRMatrix(X)) {
        X.forEachNonZeroInRow(row, callback);
        return;
    }
    for (let column = 0; column < X[row].length; column++) {
        const value = X[row][column];
        if (value !== 0) callback(column, value);
    }
}

registerSerializableClass('CSRMatrix', CSRMatrix as new (...args: never[]) => object);
