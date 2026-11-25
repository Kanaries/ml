import { kmeansIterations } from '../kmeansIterations';

test('each iteration plots a cluster label for every input point', () => {
    const points = [[20, 20], [21, 20], [60, 60], [61, 60], [100, 100], [101, 100]];
    const rows = kmeansIterations(points);
    expect(rows).toHaveLength(6 * points.length);
    for (let iterate = 1; iterate <= 6; iterate++) {
        const frame = rows.filter(row => row.iterate === iterate);
        expect(frame.map(({ x, y }) => [x, y])).toEqual(points);
        expect(frame.map(row => row.group)).toEqual([0, 0, 1, 1, 2, 2]);
    }
});
