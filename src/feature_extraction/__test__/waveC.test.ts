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
