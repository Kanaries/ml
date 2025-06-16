import { SparsePCA } from '../sparsePCA';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/sparse_pca.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const spca = new SparsePCA({ nComponents: 5, alpha: 0 });
    spca.fit(data.X);
    const pred = spca.transform(data.X_test);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            expect(pred[i][j]).toBeCloseTo(data.expected[i][j], 1);
        }
    }
});
