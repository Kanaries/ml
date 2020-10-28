import { DecisionTreeClassifier } from '../decisionTreeClassifier';

test('dtree init', () => {
    const dtree = new DecisionTreeClassifier();
    expect(dtree).toBeDefined();
})

test('basic case', () => {
    const X = [[0, 0], [1, 1], [0, 0.1], [0.1, 1], [0.9, 0.8], [1, 1.2], [1, 0]];
    const Y = [0, 1, 0, 1, 1, 1, 0];
    const dtree = new DecisionTreeClassifier();
    dtree.fit(X, Y);
    const ans = dtree.predict([[2, 2], [-1, -1]])
    expect(ans).toEqual([1, 0]);
})

test('toy sample',  () => {
    const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    const y = [-1, -1, -1, 1, 1, 1]
    const T = [[-1, -1], [2, 2], [3, 2]]
    const true_result = [-1, 1, 1]

    const dtree = new DecisionTreeClassifier();
    dtree.fit(X, y);
    const ans = dtree.predict(T);
    expect(ans).toEqual(true_result);
})