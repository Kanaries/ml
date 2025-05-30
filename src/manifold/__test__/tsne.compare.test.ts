import { TSNE } from '../tsne';
import fs from 'fs';
import path from 'path';

test('basic shape', () => {
    const p = path.join(__dirname, '../../../test_data/tsne.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const tsne = new TSNE({ nComponents: 2, perplexity: 20, learningRate: 200, nIter: 250 });
    const Y = tsne.fitTransform(data.X);
    expect(Y.length).toBe(data.X.length);
    expect(Y[0].length).toBe(2);
    for (let i = 0; i < Y.length; i++) {
        for (let j = 0; j < Y[i].length; j++) {
            expect(Number.isFinite(Y[i][j])).toBe(true);
        }
    }
});
