import { SpectralEmbedding } from '../spectralEmbedding';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/spectral_embedding.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const se = new SpectralEmbedding({ nComponents: 2, nNeighbors: 5 });
    const pred = se.fitTransform(data.X);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            // allow sign flipping
            const a = Math.abs(pred[i][j]);
            const b = Math.abs(data.expected[i][j]);
            expect(Math.abs(a - b)).toBeLessThan(0.2);
        }
    }
});
