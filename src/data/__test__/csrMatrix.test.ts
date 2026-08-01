import { decodeValue, encodeValue } from '../../base/estimator';
import { CSRMatrix, forEachNonZeroInRow, matrixRow, matrixShape } from '../csrMatrix';

describe('CSRMatrix', () => {
    const dense = [
        [0, 2, 0, 4],
        [1, 0, 0, 0],
        [0, 0, 3, 0],
    ];

    it('round-trips dense data and exposes row helpers', () => {
        const csr = CSRMatrix.fromDense(dense);
        expect(csr.shape).toEqual([3, 4]);
        expect(csr.nnz).toBe(4);
        expect(csr.toDense()).toEqual(dense);
        expect(matrixShape(csr)).toEqual([3, 4]);
        expect(matrixRow(csr, 1)).toEqual(dense[1]);
        const values: Array<[number, number]> = [];
        forEachNonZeroInRow(csr, 0, (column, value) => values.push([column, value]));
        expect(values).toEqual([[1, 2], [3, 4]]);
    });

    it('selects rows without densifying', () => {
        const selected = CSRMatrix.fromDense(dense).selectRows([2, 0]);
        expect(selected).toBeInstanceOf(CSRMatrix);
        expect(selected.toDense()).toEqual([dense[2], dense[0]]);
    });

    it('survives the model-state codec', () => {
        const csr = CSRMatrix.fromDense(dense);
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(csr)))) as CSRMatrix;
        expect(revived).toBeInstanceOf(CSRMatrix);
        expect(revived.toDense()).toEqual(dense);
    });

    it('rejects malformed CSR layouts', () => {
        expect(() => new CSRMatrix([1], [3], [0, 1], [1, 3])).toThrow(/out of bounds/);
        expect(() => new CSRMatrix([1, 2], [1, 1], [0, 2], [1, 3])).toThrow(/strictly increasing/);
        expect(() => new CSRMatrix([0], [0], [0, 1], [1, 1])).toThrow(/explicit zero/);
    });

    it('does not expose mutable backing arrays', () => {
        const csr = CSRMatrix.fromDense(dense);
        const data = csr.data as number[];
        const indices = csr.indices as number[];
        const indptr = csr.indptr as number[];
        data[0] = 99;
        indices[0] = 0;
        indptr[1] = 0;
        expect(csr.toDense()).toEqual(dense);
    });
});
