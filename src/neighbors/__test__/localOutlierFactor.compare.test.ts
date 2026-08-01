import { LocalOutlierFactor } from '../localOutlierFactor';

test('LocalOutlierFactor isolates the sklearn fixture outlier', () => {
    const X = [[0, 0], [.1, .2], [.2, .1], [1, 1], [1.1, .9], [8, 8]];
    const model = new LocalOutlierFactor({ nNeighbors: 2, contamination: 1 / 6 });
    expect(model.fitPredict(X)).toEqual([1, 1, 1, 1, 1, -1]);
    [-1, -1, -1, -3.1925824025866993, -3.1925824025866993, -8.221502494222102]
        .forEach((value, i) => expect(model.negativeOutlierFactor[i]).toBeCloseTo(value, 8));
    expect(model.offset).toBeCloseTo(-4.0307357511926005, 8);
    expect(() => model.predict([[0, 0]])).toThrow(/novelty=true/);
    expect(() => model.scoreSamples([[0, 0]])).toThrow(/novelty=true/);
});

test('LocalOutlierFactor separates novelty inference from training fitPredict', () => {
    const X = [[0, 0], [.1, .2], [.2, .1], [1, 1], [1.1, .9], [8, 8]];
    const model = new LocalOutlierFactor({ nNeighbors: 2, contamination: 1 / 6, novelty: true });
    model.fit(X);
    expect(() => model.fitPredict(X)).toThrow(/novelty=true/);
    expect(model.predict([[.15, .15], [20, 20]])).toEqual([1, -1]);
    expect(model.scoreSamples([[.15, .15]])[0]).toBeGreaterThan(model.scoreSamples([[20, 20]])[0]);
});
