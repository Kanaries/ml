import { PCA } from '../pca';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/pca.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const pca = new PCA(2);
    pca.fit(data.X);
    const pred = pca.transform(data.X_test);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            expect(pred[i][j]).toBeCloseTo(data.expected[i][j], 1);
        }
    }
});
