import { MeanShift } from '../meanShift';
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
    const p = path.join(__dirname, '../../../test_data/mean_shift.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ms = new MeanShift(2);
    const pred = ms.fitPredict(data.X);
    expect(normalize(pred)).toEqual(normalize(data.expected));
});
