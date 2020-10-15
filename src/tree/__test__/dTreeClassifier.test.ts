import { DecisionTreeClassifier } from '../decisionTreeClassifier';

test('dtree init', () => {
    const dtree = new DecisionTreeClassifier();
    expect(dtree).toBeDefined();
})

// test('basic case', () => {
//     const X = [[0, 0], [1, 1]];
//     const Y = [0, 1];
//     const dtree = new DecisionTreeClassifier();
//     dtree.fit(X, Y);
//     const ans = dtree.predict([[2, 2]])
//     expect(ans).toEqual([1]);
// })