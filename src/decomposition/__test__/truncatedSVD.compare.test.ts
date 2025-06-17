import { TruncatedSVD } from '../truncatedSVD';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/truncated_svd.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const svd = new TruncatedSVD(5);
    svd.fit(data.X);
    const pred = svd.transform(data.X_test);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            expect(pred[i][j]).toBeCloseTo(data.expected[i][j], 1);
        }
    }
});
