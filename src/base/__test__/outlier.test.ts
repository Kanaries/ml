import { OutlierBase } from '../outlier';
import { IsolationForest } from '../../ensemble';

class DummyOutlier extends OutlierBase {
    public fit(): void {}
    public predict(samplesX: number[][]): number[] {
        return samplesX.map(() => 1);
    }
}

describe('OutlierBase.fitPredict', () => {
    test('returns the predictions', () => {
        const model = new DummyOutlier();
        expect(model.fitPredict([[0], [1]])).toEqual([1, 1]);
    });

    test('IsolationForest.fitPredict returns an array of labels', () => {
        const X: number[][] = [];
        for (let i = 0; i < 30; i++) X.push([i % 5, (i * 7) % 5]);
        X.push([100, 100]);
        const labels = new IsolationForest(32, 20).fitPredict(X);
        expect(Array.isArray(labels)).toBe(true);
        expect(labels.length).toBe(X.length);
    });
});
