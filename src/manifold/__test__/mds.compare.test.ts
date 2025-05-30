import { MDS } from '../mds';
import fs from 'fs';
import path from 'path';

test('compare with reference', () => {
    const p = path.join(__dirname, '../../../test_data/mds.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const mds = new MDS({ nComponents: 2 });
    const Y = mds.fitTransform(data.X);
    const pairwise = (A: number[][]) => {
        const n = A.length;
        const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < i; j++) {
                let s = 0;
                for (let k = 0; k < A[i].length; k++) {
                    const d = A[i][k] - A[j][k];
                    s += d * d;
                }
                const v = Math.sqrt(s);
                D[i][j] = v;
                D[j][i] = v;
            }
        }
        return D;
    };
    const D1 = pairwise(Y);
    const D2 = pairwise(data.expected);
    for (let i = 0; i < D1.length; i++) {
        for (let j = 0; j < D1.length; j++) {
            expect(D1[i][j]).toBeCloseTo(D2[i][j], 2);
        }
    }
});
