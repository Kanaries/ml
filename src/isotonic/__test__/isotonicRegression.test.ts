import { IsotonicRegression } from '../isotonicRegression';

const X_DOCS = [1, 2, 3, 4, 5, 6];
const Y_DOCS = [1, 2, 3, 4, 3, 6];

describe('IsotonicRegression (PAVA)', () => {
    it('reproduces the sklearn docs case: pools the [4, 3] violation into [3.5, 3.5]', () => {
        const iso = new IsotonicRegression();
        iso.fit(X_DOCS, Y_DOCS);
        expect(iso.predict(X_DOCS)).toEqual([1, 2, 3, 3.5, 3.5, 6]);
    });

    it('interpolates linearly between thresholds', () => {
        const iso = new IsotonicRegression();
        iso.fit(X_DOCS, Y_DOCS);
        const mid = iso.predict([1.5, 3.5, 4.5, 5.5]);
        expect(mid[0]).toBeCloseTo(1.5, 12);
        expect(mid[1]).toBeCloseTo(3.25, 12);
        expect(mid[2]).toBeCloseTo(3.5, 12);
        expect(mid[3]).toBeCloseTo(4.75, 12);
    });

    it('accepts a single-column matrix and rejects multi-column input', () => {
        const iso = new IsotonicRegression();
        iso.fit(X_DOCS.map((v) => [v]), Y_DOCS);
        expect(iso.predict([[4], [4.5]])).toEqual([3.5, 3.5]);
        expect(() => iso.fit([[1, 2], [3, 4]], [1, 2])).toThrow(/1-D/);
        expect(() => iso.predict([[1, 2]])).toThrow(/1-D/);
    });

    it('transform is an alias of predict', () => {
        const iso = new IsotonicRegression();
        iso.fit(X_DOCS, Y_DOCS);
        expect(iso.transform([3.5])).toEqual(iso.predict([3.5]));
    });

    it('fits a non-increasing function when increasing=false', () => {
        const iso = new IsotonicRegression({ increasing: false });
        iso.fit([1, 2, 3, 4, 5, 6], [10, 9, 10, 7, 6, 5]);
        expect(iso.predict([1, 2, 3, 4, 5, 6])).toEqual([10, 9.5, 9.5, 7, 6, 5]);
        expect(iso.increasingFitted).toBe(false);
    });

    it("increasing='auto' detects the direction via Spearman correlation", () => {
        const up = new IsotonicRegression({ increasing: 'auto' });
        up.fit([1, 2, 3, 4, 5], [1, 3, 2, 5, 4]);
        expect(up.increasingFitted).toBe(true);

        const down = new IsotonicRegression({ increasing: 'auto' });
        down.fit([1, 2, 3, 4, 5], [10, 8, 9, 4, 2]);
        expect(down.increasingFitted).toBe(false);
        const pred = down.predict([1, 2, 3, 4, 5]);
        for (let i = 1; i < pred.length; i++) {
            expect(pred[i]).toBeLessThanOrEqual(pred[i - 1]);
        }
    });

    it('averages ties in x before PAVA (sklearn _make_unique), honoring sample weights', () => {
        const iso = new IsotonicRegression();
        iso.fit([1, 1, 2], [0, 4, 2]);
        // unweighted tie average: (0 + 4) / 2 = 2 -> constant fit
        expect(iso.predict([1, 1.5, 2])).toEqual([2, 2, 2]);

        const weighted = new IsotonicRegression();
        weighted.fit([1, 1, 2], [0, 4, 2], [3, 1, 1]);
        // weighted tie average: (0 * 3 + 4 * 1) / 4 = 1
        expect(weighted.predict([1, 2])).toEqual([1, 2]);
        expect(weighted.predict([1.5])[0]).toBeCloseTo(1.5, 12);
    });

    it('clips the fitted values to [yMin, yMax]', () => {
        const iso = new IsotonicRegression({ yMin: 2, yMax: 3.6 });
        iso.fit(X_DOCS, Y_DOCS);
        expect(iso.predict(X_DOCS)).toEqual([2, 2, 3, 3.5, 3.5, 3.6]);
    });

    it("outOfBounds='clip' (default) evaluates at the boundaries", () => {
        const iso = new IsotonicRegression();
        iso.fit(X_DOCS, Y_DOCS);
        expect(iso.predict([0, 100])).toEqual([1, 6]);
    });

    it("outOfBounds='nan' returns NaN outside the training range", () => {
        const iso = new IsotonicRegression({ outOfBounds: 'nan' });
        iso.fit(X_DOCS, Y_DOCS);
        const out = iso.predict([0, 3, 100]);
        expect(out[0]).toBeNaN();
        expect(out[1]).toBe(3);
        expect(out[2]).toBeNaN();
    });

    it("outOfBounds='raise' throws outside the training range", () => {
        const iso = new IsotonicRegression({ outOfBounds: 'raise' });
        iso.fit(X_DOCS, Y_DOCS);
        expect(() => iso.predict([0])).toThrow(/outside the training range/);
        expect(() => iso.predict([7])).toThrow(/outside the training range/);
        expect(iso.predict([1, 6])).toEqual([1, 6]);
    });

    it('validates constructor params and fit inputs', () => {
        expect(() => new IsotonicRegression({ increasing: 'up' as never })).toThrow(/increasing/);
        expect(() => new IsotonicRegression({ outOfBounds: 'wrap' as never })).toThrow(/outOfBounds/);
        expect(() => new IsotonicRegression({ yMin: 2, yMax: 1 })).toThrow(/yMin must be <= yMax/);
        const iso = new IsotonicRegression();
        expect(() => iso.fit([1, 2], [1])).toThrow(/same length/);
        expect(() => iso.fit([1, 2], [1, 2], [1])).toThrow(/sampleWeight/);
        expect(() => iso.fit([1, 2], [1, 2], [1, -1])).toThrow(/sampleWeight/);
        expect(() => iso.predict([1])).toThrow(/must be fitted/);
    });
});
