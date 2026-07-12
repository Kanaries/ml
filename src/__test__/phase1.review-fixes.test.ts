/**
 * Regression tests for the CodeX Phase 1 review findings
 * (docs/reviews/phase1-codex.md). Each block corresponds to one finding.
 */
import { crossValidate, StratifiedShuffleSplit } from '../utils/modelSelection';
import { QuantileTransformer, PowerTransformer, KNNImputer } from '../utils/preprocessing';
import { trainTestSplit } from '../utils/sampling';
import { ColumnTransformer } from '../pipeline';
import { StandardScaler } from '../utils/preprocessing';
import { rocAucScore } from '../metrics';
import { LogisticRegression } from '../linear/logisticRegression';

describe('P1: classifiers get stratified default CV', () => {
    it('crossValidate succeeds on label-ordered data that plain KFold would break', () => {
        // labels sorted by class: KFold(2) would put all of class 0 in one fold
        const X = Array.from({ length: 12 }, (_, i) => [i < 6 ? i : 100 + i]);
        const y = X.map((_, i) => (i < 6 ? 0 : 1));
        const result = crossValidate(new LogisticRegression({ maxIter: 100 }), X, y, { cv: 2 });
        // stratified folds keep both classes in every training fold → the
        // (perfectly separable) problem scores 1.0 everywhere
        expect(result.testScore.score).toEqual([1, 1]);
    });
});

describe('P1: StratifiedShuffleSplit randomizes remainder allocation per split', () => {
    it('does not systematically give the same class the extra test sample', () => {
        // 3 equal classes, test size 4 → one class gets 2 test samples per split
        const X = Array.from({ length: 30 }, (_, i) => [i]);
        const y = X.map((_, i) => i % 3);
        const splitter = new StratifiedShuffleSplit({ nSplits: 40, testSize: 4, randomState: 7 });
        const overRepresented = new Set<number>();
        for (const fold of splitter.split(X, y)) {
            const counts = [0, 0, 0];
            for (const i of fold.testIndices) counts[y[i]]++;
            overRepresented.add(counts.indexOf(2));
        }
        // over 40 seeded splits every class should get the extra slot at least once
        expect(overRepresented.size).toBe(3);
    });
});

describe('P1: ColumnTransformer rejects schema drift at transform time', () => {
    it('throws when inference data has a different column count', () => {
        const ct = new ColumnTransformer({ transformers: [['s', new StandardScaler(), [0]]] });
        ct.fit([[1, 2, 3], [4, 5, 6]]);
        expect(() => ct.transform([[1, 2]])).toThrow(/fitted with 3/);
        expect(() => ct.transform([[1, 2, 3, 4]])).toThrow(/fitted with 3/);
    });

    it('rejects transformer names containing __', () => {
        expect(() => new ColumnTransformer({
            transformers: [['scale__numeric', new StandardScaler(), [0]]],
        })).toThrow(/__/);
    });
});

describe('P1: PowerTransformer lambda search expands beyond [-5, 5]', () => {
    it('finds strongly negative lambdas for near-constant positive data', () => {
        // sklearn: PowerTransformer('box-cox').fit([[1],[1.001],[1.002],[1.1]])
        // → lambda ≈ -37.8; a [-5,5]-clamped search would return -5
        const pt = new PowerTransformer({ method: 'box-cox', standardize: false });
        pt.fit([[1], [1.001], [1.002], [1.1]]);
        const lambda = (pt as unknown as { lambdas: number[] }).lambdas[0];
        expect(lambda).toBeLessThan(-20);
    });
});

describe('P1: KNNImputer tolerates all-missing training features', () => {
    it('drops the empty feature like sklearn instead of throwing', () => {
        const imputer = new KNNImputer({ nNeighbors: 1 });
        imputer.fit([[1, NaN], [2, NaN]]);
        // output drops the all-missing column
        expect(imputer.transform([[NaN, 5], [3, 6]])).toEqual([[1.5], [3]]);
    });
});

describe('P2: metric and size validation', () => {
    it('rocAucScore throws when only one class is present', () => {
        expect(() => rocAucScore([1, 1], [0.2, 0.8])).toThrow(/one class/);
    });

    it('QuantileTransformer rejects nQuantiles > subsample', () => {
        expect(() => new QuantileTransformer({ nQuantiles: 100, subsample: 10 })).toThrow(/subsample/);
    });

    it('trainTestSplit rejects non-integer absolute sizes', () => {
        const X = Array.from({ length: 10 }, (_, i) => [i]);
        const y = X.map((_, i) => i % 2);
        expect(() => trainTestSplit(X, y, { testSize: 1.9 })).toThrow(/integer/);
        expect(() => trainTestSplit(X, y, { trainSize: 5.5 })).toThrow(/integer/);
    });
});
