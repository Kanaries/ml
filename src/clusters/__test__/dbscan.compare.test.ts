import { DBScan } from '../dbscan';
import fs from 'fs';
import path from 'path';

function normalize(labels: number[]): number[] {
    const map = new Map<number, number>();
    let counter = 0;
    return labels.map(l => {
        if (!map.has(l)) map.set(l, counter++);
        return map.get(l)!;
    });
}

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/dbscan.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const db = new DBScan(0.8, 3);
    const pred = db.fitPredict(data.X);
    expect(normalize(pred)).toEqual(normalize(data.expected));
});
