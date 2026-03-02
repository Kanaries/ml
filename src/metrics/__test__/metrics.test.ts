import {
    accuracyScore,
    precisionScore,
    recallScore,
    f1Score,
    meanSquaredError,
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
