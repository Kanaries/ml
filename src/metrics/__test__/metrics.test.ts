import {
    accuracyScore,
    adjustedRandScore,
    confusionMatrix,
    precisionScore,
    precisionRecallCurve,
    precisionRecallFscoreSupport,
    recallScore,
    f1Score,
    meanSquaredError,
    rocAucScore,
    rocCurve,
    r2Score,
} from '../index';

test('accuracyScore supports normalized and raw counts', () => {
    const actual = [1, 0, 1, 1];
    const expected = [1, 1, 1, 0];
    expect(accuracyScore(actual, expected)).toBeCloseTo(0.5);
    expect(accuracyScore(actual, expected, false)).toBe(2);
});

test('precision, recall, and f1 work for binary classification', () => {
    const actual = [1, 1, 0, 0, 1];
    const expected = [1, 0, 0, 0, 1];

    expect(precisionScore(actual, expected)).toBeCloseTo(2 / 3);
    expect(recallScore(actual, expected)).toBeCloseTo(1);
    expect(f1Score(actual, expected)).toBeCloseTo(0.8);
});

test('precision, recall, and f1 support macro average', () => {
    const actual = [0, 1, 2, 2, 1, 0];
    const expected = [0, 2, 2, 1, 1, 0];

    expect(precisionScore(actual, expected, { average: 'macro' })).toBeCloseTo(2 / 3);
    expect(recallScore(actual, expected, { average: 'macro' })).toBeCloseTo(2 / 3);
    expect(f1Score(actual, expected, { average: 'macro' })).toBeCloseTo(2 / 3);
});

test('meanSquaredError and r2Score for regression', () => {
    const expected = [3, -0.5, 2, 7];
    const actual = [2.5, 0, 2, 8];

    expect(meanSquaredError(actual, expected)).toBeCloseTo(0.375);
    expect(r2Score(actual, expected)).toBeCloseTo(0.948608137);
});

test('metrics validate input lengths', () => {
    expect(() => accuracyScore([1], [1, 0])).toThrow('actual and expected must have the same length');
    expect(() => meanSquaredError([], [])).toThrow('actual and expected must be non-empty');
});

test('precisionRecallFscoreSupport returns macro metrics and supports', () => {
    const actual = [0, 1, 2, 2, 1, 0];
    const expected = [0, 2, 2, 1, 1, 0];

    const result = precisionRecallFscoreSupport(actual, expected, { average: 'macro' });
    expect(result.precision).toBeCloseTo(2 / 3);
    expect(result.recall).toBeCloseTo(2 / 3);
    expect(result.fScore).toBeCloseTo(2 / 3);
    expect(result.support).toEqual([2, 2, 2]);
});

test('confusionMatrix counts predictions by class order', () => {
    const actual = [0, 1, 0, 1];
    const expected = [0, 0, 1, 1];

    expect(confusionMatrix(actual, expected)).toEqual([
        [1, 1],
        [1, 1],
    ]);
});

test('rocCurve, precisionRecallCurve, and rocAucScore compute binary ranking metrics', () => {
    const scores = [0.1, 0.4, 0.35, 0.8];
    const expected = [0, 0, 1, 1];

    const roc = rocCurve(expected, scores);
    expect(roc.fpr[0]).toBe(0);
    expect(roc.tpr[0]).toBe(0);
    expect(roc.thresholds[0]).toBe(Infinity);
    expect(rocAucScore(expected, scores)).toBeCloseTo(0.75);

    const pr = precisionRecallCurve(expected, scores);
    expect(pr.precision[0]).toBeCloseTo(2 / 4);
    expect(pr.recall[0]).toBe(1);
    expect(pr.thresholds).toEqual([0.1, 0.35, 0.4, 0.8]);
});

test('adjustedRandScore handles perfect and imperfect clustering matches', () => {
    expect(adjustedRandScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1);
    expect(adjustedRandScore([0, 0, 1, 1], [0, 1, 0, 1])).toBeCloseTo(-0.5);
});

describe('macro f1 (sklearn definition: mean of per-class F1)', () => {
    test('matches hand-computed sklearn value', () => {
        const actual = [1, 0, 0]; // predictions
        const expected = [1, 1, 0]; // ground truth
        // class 1: P=1, R=0.5, F1=2/3; class 0: P=0.5, R=1, F1=2/3 -> macro F1 = 2/3
        expect(f1Score(actual, expected, { average: 'macro' })).toBeCloseTo(2 / 3, 10);
    });

    test('agrees with precisionRecallFscoreSupport', () => {
        const actual = [1, 0, 0, 2, 2, 1];
        const expected = [1, 1, 0, 2, 1, 2];
        const { fScore } = precisionRecallFscoreSupport(actual, expected, { average: 'macro' });
        expect(f1Score(actual, expected, { average: 'macro' })).toBeCloseTo(fScore, 10);
    });
});
