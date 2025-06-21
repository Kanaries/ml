import { LinearSVR } from '../linearSVR';

test('basic', () => {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 20; i++) {
        const x = i;
        X.push([x]);
        y.push(2 * x + 1);
    }
    const reg = new LinearSVR({ learningRate: 0.01, maxIter: 200 });
    reg.fit(X, y);
    const pred = reg.predict([[25], [30]]);
    expect(Math.abs(pred[0] - (2 * 25 + 1))).toBeLessThan(2);
    expect(Math.abs(pred[1] - (2 * 30 + 1))).toBeLessThan(2);
});
