import fs from 'fs';
import path from 'path';
import { DictVectorizer } from '../dictVectorizer';
import { FeatureHasher } from '../featureHasher';
import { HashingVectorizer } from '../hashingVectorizer';
import { CSRMatrix } from '../../data';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

test('HashingVectorizer matches sklearn murmurhash indices and signs', () => {
    const fixture = waveC.hashing_vectorizer, model = new HashingVectorizer({ nFeatures: 16, norm: null, alternateSign: true });
    expect(model.fitTransform(fixture.documents).toDense()).toEqual(fixture.matrix);
});

test('HashingVectorizer default l2 normalization stays sparse and matches sklearn', () => {
    const fixture = waveC.hashing_vectorizer, vectorizer = new HashingVectorizer({ nFeatures: 16 });
    const denseSpy = jest.spyOn(CSRMatrix.prototype, 'toDense').mockImplementation(() => { throw new Error('unexpected densification'); });
    const matrix = vectorizer.transform(fixture.documents); denseSpy.mockRestore();
    matrix.toDense().forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.default_matrix[i][j], 12)));
});

test('DictVectorizer matches sklearn categorical expansion and sorted vocabulary', () => {
    const fixture = waveC.dict_vectorizer, model = new DictVectorizer({ sparse: false }); model.fit(fixture.rows);
    expect(model.getFeatureNamesOut()).toEqual(fixture.feature_names); expect(model.transform(fixture.rows)).toEqual(fixture.matrix);
    expect(model.inverseTransform(fixture.matrix)).toEqual(fixture.matrix.map((row: number[]) => Object.fromEntries(row.flatMap((value, i) => value === 0 ? [] : [[fixture.feature_names[i], value]]))));
    model.restrict([true, false, true]); expect(model.getFeatureNamesOut()).toEqual([fixture.feature_names[0], fixture.feature_names[2]]); expect(model.transform(fixture.rows)).toEqual(fixture.matrix.map((row: number[]) => [row[0], row[2]]));
});

test('FeatureHasher matches sklearn dict hashing', () => {
    const fixture = waveC.feature_hasher, model = new FeatureHasher({ nFeatures: 16, inputType: 'dict', alternateSign: true });
    expect(model.fitTransform(fixture.rows).toDense()).toEqual(fixture.matrix);
});

test.each(waveC.feature_hasher_cases)(
    'FeatureHasher matches sklearn for $inputType inputs with alternateSign=$alternateSign',
    ({ inputType, alternateSign, rows, matrix }) => {
        const model = new FeatureHasher({ nFeatures: 4, inputType, alternateSign });
        const result = model.fitTransform(rows) as CSRMatrix;
        expect(result).toBeInstanceOf(CSRMatrix);
        expect(result.shape).toEqual([rows.length, 4]);
        expect(result.toDense()).toEqual(matrix);
        expect(result.data).not.toContain(0);
    },
);

test('dict string values hash as categorical features', () => {
    const options = { nFeatures: 16 };
    const actual = new FeatureHasher(options).transform([{ city: 'London', temp: 12 }]);
    const encoded = new FeatureHasher({ ...options, inputType: 'pair' }).transform([[['city=London', 1], ['temp', 12]]]);
    expect(actual.toDense()).toEqual(encoded.toDense());
});

test('FeatureHasher rejects malformed samples and invalid dimensions', () => {
    const dict = new FeatureHasher();
    for (const sample of [null, [], { key: Infinity }, { key: false }]) {
        expect(() => dict.transform([sample])).toThrow();
    }
    const tokens = new FeatureHasher({ inputType: 'string' });
    expect(() => tokens.transform(['abc'])).toThrow(/arrays/);
    expect(() => tokens.transform([['abc', 2]])).toThrow(/strings/);
    const pairs = new FeatureHasher({ inputType: 'pair' });
    for (const pair of ['x', ['x'], [2, 1], ['x', '1'], ['x', NaN]]) {
        expect(() => pairs.transform([[pair]])).toThrow(/string, number/);
    }
    for (const nFeatures of [0, 1.5]) {
        expect(() => new FeatureHasher({ nFeatures })).toThrow(/parameters/);
    }
});
