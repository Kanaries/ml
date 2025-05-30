import { LocallyLinearEmbedding } from '../lle';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/lle.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const lle = new LocallyLinearEmbedding(5, 2);
    lle.fit(data.X);
    const pred = lle.transform(data.X_test);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            expect(Math.abs(pred[i][j])).toBeCloseTo(Math.abs(data.expected[i][j]), 1);
        }
    }
});
