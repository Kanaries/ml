import { ExtraTreeClassifier } from '../extraTreeClassifier';
import { loadModel, SerializedModel } from '../../base';

test('extra tree init', () => {
    const clf = new ExtraTreeClassifier();
    expect(clf).toBeDefined();
});

test('basic case', () => {
    const X = [[0, 0], [1, 1], [0, 0.1], [0.1, 1], [0.9, 0.8], [1, 1.2], [1, 0]];
    const Y = [0, 1, 0, 1, 1, 1, 0];
    const clf = new ExtraTreeClassifier();
    clf.fit(X, Y);
    const ans = clf.predict([[2, 2], [-1, -1]]);
    expect(ans.length).toBe(2);
});

test('is deterministic with randomState', () => {
    const X = [[0, 0], [1, 1], [0.1, 0.2], [0.9, 0.8], [1.2, 1.1], [-0.2, 0.1]];
    const Y = [0, 1, 0, 1, 1, 0];
    const clf1 = new ExtraTreeClassifier({ randomState: 7 });
    const clf2 = new ExtraTreeClassifier({ randomState: 7 });
    clf1.fit(X, Y);
    clf2.fit(X, Y);
    const testX = [[0.8, 0.9], [0, 0], [1.1, 1.0]];
    expect(clf1.predict(testX)).toEqual(clf2.predict(testX));
});

test('pre-probability serialized trees retain their legacy predictions', () => {
    const X = [[0, 0], [1, 1], [0.1, 0.2], [0.9, 0.8], [1.2, 1.1], [-0.2, 0.1]];
    const y = [0, 1, 0, 1, 1, 0];
    const model = new ExtraTreeClassifier({ max_depth: 2, randomState: 7 });
    model.fit(X, y);
    const expected = model.predict(X);
    const legacy = JSON.parse(JSON.stringify(model.toJSON())) as SerializedModel;
    const state = legacy.state as Record<string, unknown>;
    delete state.classesState;
    const removeProbabilities = (node: Record<string, unknown> | null): void => {
        if (!node) return;
        delete node.classProbabilities;
        removeProbabilities(node.leftChild as Record<string, unknown> | null);
        removeProbabilities(node.rightChild as Record<string, unknown> | null);
    };
    removeProbabilities(state.dtree as Record<string, unknown>);
    const revived = loadModel(legacy) as ExtraTreeClassifier;
    expect(revived.predict(X)).toEqual(expected);
    revived.predictProba(X).forEach(row => expect(row.reduce((sum, value) => sum + value, 0)).toBe(1));
});
