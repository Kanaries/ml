import { lstsq } from '../lstsq';
import { gaussianElimination } from '../basic';

describe('lstsq (Householder QR)', () => {
    test('solves an exactly determined system', () => {
        // x + y = 3, x - y = 1 -> x = 2, y = 1
        const w = lstsq([[1, 1], [1, -1]], [3, 1]);
        expect(w).not.toBe(false);
        expect((w as number[])[0]).toBeCloseTo(2, 10);
        expect((w as number[])[1]).toBeCloseTo(1, 10);
    });

    test('solves an overdetermined system in the least-squares sense', () => {
        // y = 2x + 1 with a perturbed point; lstsq of [1, x]
        const A = [[1, 0], [1, 1], [1, 2], [1, 3]];
        const b = [1, 3, 5, 7.4];
        const w = lstsq(A, b) as number[];
        // normal equations by hand: [[4,6],[6,14]] w = [16.4, 35.2] -> [0.92, 2.12]
        expect(w[0]).toBeCloseTo(0.92, 10);
        expect(w[1]).toBeCloseTo(2.12, 10);
    });

    test('returns false on exactly collinear columns', () => {
        expect(lstsq([[1, 1], [2, 2], [3, 3]], [1, 2, 3])).toBe(false);
    });

    test('returns false when n_samples < n_features', () => {
        expect(lstsq([[1, 2, 3]], [1])).toBe(false);
    });

    test('handles ill-conditioned but full-rank systems without garbage', () => {
        const A = [
            [1, 1.0000001],
            [2, 2.0000002],
            [3, 3.0000004],
        ];
        const b = [1, 2, 3];
        const w = lstsq(A, b) as number[];
        expect(w).not.toBe(false);
        const pred = 4 * w[0] + 4 * w[1];
        expect(Math.abs(pred - 4)).toBeLessThan(0.01);
    });
});

describe('gaussianElimination pivoting', () => {
    test('partial pivoting keeps tiny-pivot systems accurate', () => {
        // without |value| pivoting, the 1e-20 pivot destroys all precision
        const res = gaussianElimination([
            [1e-20, 1, 1],
            [1, 1, 2],
        ]);
        expect(res).not.toBe(false);
        // solution of the augmented system: y = 1, x = 1 (to double precision)
        const rows = res as number[][];
        // back-substitute: second row gives y, first row gives x
        const y = rows[1][2] / rows[1][1];
        const x = (rows[0][2] - rows[0][1] * y) / rows[0][0];
        expect(y).toBeCloseTo(1, 8);
        expect(x).toBeCloseTo(1, 8);
    });

    test('detects numerically singular matrices instead of dividing by ~0', () => {
        // rank-1 matrix with float noise far below the matrix scale
        const singular = gaussianElimination([
            [1, 2],
            [0.5, 1 + 1e-17],
        ]);
        expect(singular).toBe(false);
    });
});
