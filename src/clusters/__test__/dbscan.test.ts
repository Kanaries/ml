import { DBScan } from '../dbscan';

function makeCircles(n1: number, n2: number, r1: number, r2: number): number[][] {
    const data: number[][] = [];
    for (let i = 0; i < n1; i++) {
        const angle = (2 * Math.PI * i) / n1;
        data.push([r1 * Math.cos(angle), r1 * Math.sin(angle)]);
    }
    for (let j = 0; j < n2; j++) {
        const angle = (2 * Math.PI * j) / n2;
        data.push([r2 * Math.cos(angle), r2 * Math.sin(angle)]);
    }
    return data;
}

function reAssignLabel(labels: number[]): number[] {
    const encoder: Map<number, number> = new Map();
    let ans: number[] = [];
    let counter = 0;
    for (let i = 0; i < labels.length; i++) {
        if (!encoder.has(labels[i])) {
            encoder.set(labels[i], counter++);
        }
        ans.push(encoder.get(labels[i])!);
    }
    return ans;
}

test('dbscan two circles', () => {
    const X = makeCircles(20, 20, 1, 5);
    const dbscan = new DBScan(0.6, 3);
    const labels = reAssignLabel(dbscan.fitPredict(X));
    const expected: number[] = [];
    for (let i = 0; i < 20; i++) expected.push(0);
    for (let i = 0; i < 20; i++) expected.push(1);
    expect(labels).toEqual(expected);
});
