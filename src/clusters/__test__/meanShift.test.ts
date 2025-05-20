import { MeanShift } from '../meanShift';

function reAssignLabel (labels: number[]): number[] {
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

test('init', () => {
    const ms = new MeanShift();
    expect(ms).toBeDefined();
});

test('meanshift simple clusters', () => {
    const X = [
        [0, 0],
        [0.1, 0],
        [-0.1, 0],
        [3, 3],
        [3.1, 3],
        [3, 3.1]
    ];
    const ms = new MeanShift(1);
    const labels = reAssignLabel(ms.fitPredict(X));
    const expected = [0, 0, 0, 1, 1, 1];
    expect(labels).toEqual(expected);
});
