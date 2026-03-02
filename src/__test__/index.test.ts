import * as ML from '../index';

test('top-level export provides utils namespace', () => {
    expect(ML.utils).toBeDefined();
    expect(typeof ML.utils.asyncMode).toBe('function');
    expect(typeof ML.utils.Sampling.trainTestSplit).toBe('function');
    expect(typeof ML.utils.Preprocessing.StandardScaler).toBe('function');
    expect(typeof ML.utils.ModelSelection.crossValScore).toBe('function');
    expect(typeof ML.Metrics.accuracyScore).toBe('function');
});
