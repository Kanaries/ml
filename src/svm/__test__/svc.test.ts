import { SVC } from '../svc';
import { LinearSVC } from '../linearSVC';
import { NuSVC } from '../nuSVC';

test('linear svc basic', () => {
    const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]];
    const y = [0, 0, 0, 1, 1, 1];
    const T = [[-1, -1], [2, 2], [3, 2]];
    const trueResult = [0, 1, 1];

    const svc = new SVC({ kernel: 'linear', maxIter: 50, learningRate: 0.1 });
    svc.fit(X, y);
    const ans = svc.predict(T);
    expect(ans).toEqual(trueResult);
});

test('nu svc basic', () => {
    const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]];
    const y = [0, 0, 0, 1, 1, 1];
    const T = [[-1, -1], [2, 2], [3, 2]];
    const trueResult = [0, 1, 1];

    const svc = new NuSVC({ kernel: 'linear', maxIter: 50, learningRate: 0.1 });
    svc.fit(X, y);
    const ans = svc.predict(T);
    expect(ans).toEqual(trueResult);
});

test('linearSVC basic', () => {
    const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]];
    const y = [0, 0, 0, 1, 1, 1];
    const T = [[-1, -1], [2, 2], [3, 2]];
    const trueResult = [0, 1, 1];

    const svc = new LinearSVC({ maxIter: 50, learningRate: 0.1 });
    svc.fit(X, y);
    const ans = svc.predict(T);
    expect(ans).toEqual(trueResult);
});
