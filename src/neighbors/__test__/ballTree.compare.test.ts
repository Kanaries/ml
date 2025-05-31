import { BallTree } from '../ballTree';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/balltree.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const tree = new BallTree(data.X, data.leaf_size);
    const q = tree.query(data.test, data.k);
    for (let i = 0; i < data.test.length; i++) {
        for (let j = 0; j < data.k; j++) {
            expect(q.indices[i][j]).toBe(data.query_indices[i][j]);
            expect(q.distances[i][j]).toBeCloseTo(data.query_distances[i][j], 6);
        }
    }
    const r = tree.queryRadius(data.test, data.radius, true) as {
        indices: number[][];
        distances: number[][];
    };
    for (let i = 0; i < data.test.length; i++) {
        const idx = [...r.indices[i]].sort((a, b) => a - b);
        const idxExp = [...data.radius_indices[i]].sort((a, b) => a - b);
        expect(idx).toEqual(idxExp);
        const dist = [...r.distances[i]].sort((a, b) => a - b);
        const distExp = [...data.radius_distances[i]].sort((a, b) => a - b);
        for (let j = 0; j < dist.length; j++) {
            expect(dist[j]).toBeCloseTo(distExp[j], 6);
        }
    }
});
