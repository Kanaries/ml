import { OneClassSVM } from '../oneClassSVM';

// three tight 2-D blobs (60 inliers) plus three far-away outliers
const inliers: number[][] = [];
for (const [cx, cy] of [[0, 0], [4, 4], [0, 4]]) {
    for (let i = 0; i < 20; i++) {
        inliers.push([
            cx + 0.3 * Math.sin(i * 2.4 + cx),
            cy + 0.3 * Math.cos(i * 1.7 + cy),
        ]);
    }
}
const farOutliers = [[10, 10], [-8, 9], [9, -8]];
const blobX = [...inliers, ...farOutliers];

describe('OneClassSVM (SMO, Schölkopf one-class dual)', () => {
    test('rbf kernel flags far outliers with -1 and ~1-nu of the data as inliers', () => {
        const nu = 0.1;
        const oc = new OneClassSVM({ kernel: 'rbf', nu });
        oc.fit(blobX);
        const pred = oc.predict(blobX);
        // labels are the sklearn convention: +1 inlier / -1 outlier
        for (const p of pred) {
            expect([1, -1]).toContain(p);
        }
        // the three planted far outliers must be flagged
        expect(pred.slice(inliers.length)).toEqual([-1, -1, -1]);
        // nu upper-bounds the training outlier fraction and the solution
        // typically saturates it: outlier fraction ~ nu (margin SVs with
        // f = 0 are labelled -1, so allow tolerance both ways)
        const outlierFrac = pred.filter(p => p === -1).length / blobX.length;
        expect(Math.abs(outlierFrac - nu)).toBeLessThanOrEqual(0.08);
    });

    test('predict is the sign of decisionFunction', () => {
        const oc = new OneClassSVM({ kernel: 'rbf', nu: 0.2 });
        oc.fit(blobX);
        const dec = oc.decisionFunction(blobX);
        const pred = oc.predict(blobX);
        dec.forEach((d, i) => {
            expect(pred[i]).toBe(d > 0 ? 1 : -1);
        });
        // far outliers get strictly negative scores
        expect(dec[inliers.length + 0]).toBeLessThan(0);
        expect(dec[inliers.length + 1]).toBeLessThan(0);
        expect(dec[inliers.length + 2]).toBeLessThan(0);
    });

    test('linear kernel separates a blob from points near the origin', () => {
        // linear one-class separates the data from the origin with margin
        // rho/||w||, so points pulled back toward the origin are outliers
        const blob: number[][] = [];
        for (let i = 0; i < 30; i++) {
            blob.push([
                5 + 0.3 * Math.sin(i * 2.4),
                5 + 0.3 * Math.cos(i * 1.7),
            ]);
        }
        const nearOrigin = [[0.1, 0.2], [0.3, 0.05]];
        const oc = new OneClassSVM({ kernel: 'linear', nu: 0.1 });
        oc.fit([...blob, ...nearOrigin]);
        const pred = oc.predict([...blob, ...nearOrigin]);
        expect(pred.slice(blob.length)).toEqual([-1, -1]);
        const inlierRate = pred.slice(0, blob.length).filter(p => p === 1).length / blob.length;
        expect(inlierRate).toBeGreaterThanOrEqual(0.8);
    });

    test('nu upper-bounds training errors and lower-bounds the SV fraction', () => {
        const n = blobX.length;
        for (const nu of [0.1, 0.5]) {
            const oc = new OneClassSVM({ kernel: 'rbf', nu });
            oc.fit(blobX);
            const outlierFrac = oc.predict(blobX).filter(p => p === -1).length / n;
            const svFrac = oc.getSupportVectors().length / n;
            expect(outlierFrac).toBeLessThanOrEqual(nu + 2 / n);
            expect(svFrac).toBeGreaterThanOrEqual(nu - 2 / n);
        }
    });

    test('fitPredict matches fit + predict', () => {
        const a = new OneClassSVM({ kernel: 'rbf', nu: 0.1 });
        const viaFitPredict = a.fitPredict(blobX);
        const b = new OneClassSVM({ kernel: 'rbf', nu: 0.1 });
        b.fit(blobX);
        expect(viaFitPredict).toEqual(b.predict(blobX));
    });

    test('invalid nu throws, predict before fit throws', () => {
        expect(() => new OneClassSVM({ nu: 0 })).toThrow();
        expect(() => new OneClassSVM({ nu: 1.5 })).toThrow();
        expect(() => new OneClassSVM().predict([[0, 0]])).toThrow();
        expect(() => new OneClassSVM().fit([])).toThrow();
    });
});
