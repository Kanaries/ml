import { BernoulliRBM } from '../bernoulliRBM';

test('init', () => {
    const rbm = new BernoulliRBM();
    expect(rbm).toBeDefined();
});

test('basic fit transform', () => {
    const X = [
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 1]
    ];
    const rbm = new BernoulliRBM({ nComponents: 2, nIter: 5 });
    rbm.fit(X);
    const trans = rbm.transform(X);
    expect(trans.length).toBe(3);
    expect(trans[0].length).toBe(2);
});
