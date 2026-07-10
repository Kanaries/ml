import { DecisionTreeClassifier } from '../decisionTreeClassifier';
import { DecisionTreeRegressor } from '../decisionTreeRegressor';
import { ExtraTreeClassifier } from '../extraTreeClassifier';
import { ExtraTreeRegressor } from '../extraTreeRegressor';
import type { IDTree } from '../decisionTreeClassifier';

const X5 = Array.from({ length: 30 }, (_, i) => [i, i % 7, (i * 3) % 11, (i * 5) % 13, (i * 7) % 17]);
const Y5 = X5.map(row => (row[0] > 14 ? 1 : 0));

function treeDepth(t: IDTree | null): number {
    if (!t || t.splitIndex === -1) return 0;
    return 1 + Math.max(treeDepth(t.leftChild), treeDepth(t.rightChild));
}

describe('unified max_features semantics (JS cannot tell 1 from 1.0)', () => {
    test('DecisionTreeClassifier: integer 1 selects exactly one feature', () => {
        const m = new DecisionTreeClassifier({ max_features: 1, randomState: 42 });
        m.fit(X5, Y5);
        expect((m as any).selectedFeatureIndices().length).toBe(1);
    });

    test('DecisionTreeClassifier: fraction floors (0.5 of 5 -> 2)', () => {
        const m = new DecisionTreeClassifier({ max_features: 0.5, randomState: 42 });
        m.fit(X5, Y5);
        expect((m as any).selectedFeatureIndices().length).toBe(2);
    });

    test('DecisionTreeClassifier: sqrt floors like sklearn (sqrt of 5 -> 2)', () => {
        const m = new DecisionTreeClassifier({ max_features: 'sqrt', randomState: 42 });
        m.fit(X5, Y5);
        expect((m as any).selectedFeatureIndices().length).toBe(2);
    });

    test('DecisionTreeRegressor: integer 1 selects exactly one feature', () => {
        const m = new DecisionTreeRegressor({ max_features: 1, randomState: 42 });
        m.fit(X5, X5.map(r => r[0]));
        expect((m as any).selectedFeatureIndices().length).toBe(1);
    });

    test('ExtraTreeClassifier: fractional max_features must not degenerate to a constant model', () => {
        const X = Array.from({ length: 40 }, (_, i) => [i, (i * 13) % 40]);
        const Y = X.map(row => (row[0] > 19 ? 1 : 0));
        const m = new ExtraTreeClassifier({ max_features: 0.5, randomState: 7 });
        m.fit(X, Y);
        const acc = m.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBeGreaterThan(0.9);
    });

    test('ExtraTreeClassifier: max_depth=1 builds a tree with at most one split level', () => {
        const m = new ExtraTreeClassifier({ max_depth: 1, randomState: 7 });
        m.fit(X5, Y5);
        expect(treeDepth((m as any).dtree)).toBeLessThanOrEqual(1);
    });

    test('ExtraTreeRegressor: randomState makes fits reproducible', () => {
        const X = Array.from({ length: 50 }, (_, i) => [Math.sin(i) * 10, Math.cos(i * 1.7) * 10]);
        const Y = X.map(r => r[0] * 2 + r[1]);
        const a = new ExtraTreeRegressor({ randomState: 5, max_features: 1 });
        const b = new ExtraTreeRegressor({ randomState: 5, max_features: 1 });
        a.fit(X, Y);
        b.fit(X, Y);
        expect(a.predict(X)).toEqual(b.predict(X));
    });
});
