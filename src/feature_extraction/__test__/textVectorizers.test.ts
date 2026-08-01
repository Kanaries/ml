import { loadModel } from '../../base';
import fs from 'fs';
import path from 'path';
import { MultinomialNB } from '../../bayes';
import { Pipeline } from '../../pipeline';
import { CountVectorizer } from '../countVectorizer';
import { TfidfTransformer } from '../tfidfTransformer';
import { TfidfVectorizer } from '../tfidfVectorizer';

const waveB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_b.json'), 'utf8'));
const documents: string[] = waveB.text.documents;

test('CountVectorizer matches the sklearn documentation fixture', () => {
    const vectorizer = new CountVectorizer();
    const counts = vectorizer.fitTransform(documents);
    expect(vectorizer.getFeatureNamesOut()).toEqual(waveB.text.count_feature_names);
    expect(counts.toDense()).toEqual(waveB.text.count_matrix);
});

test('CountVectorizer applies ngrams, stop words, document thresholds, and binary counts', () => {
    const vectorizer = new CountVectorizer({ stopWords: ['the'], ngramRange: [1, 2], minDf: 2, binary: true });
    const counts = vectorizer.fitTransform(documents);
    expect(vectorizer.getFeatureNamesOut()).toEqual(waveB.text.options_feature_names);
    expect(counts.toDense()).toEqual(waveB.text.options_matrix);
    expect(counts.toDense().every(row => row.every(value => value === 0 || value === 1))).toBe(true);
});

test('CountVectorizer fixed vocabulary supports transform without fit', () => {
    const vectorizer = new CountVectorizer({ vocabulary: { apple: 0, blue: 1 } });
    expect(vectorizer.getFeatureNamesOut()).toEqual(['apple', 'blue']);
    expect(vectorizer.transform(['blue apple apple']).toDense()).toEqual([[2, 1]]);
});

test('TfidfTransformer matches sklearn smooth idf and returns normalized CSR', () => {
    const counts = new CountVectorizer().fitTransform(documents);
    const transformer = new TfidfTransformer();
    const tfidf = transformer.fitTransform(counts);
    const expectedIdf = waveB.text.idf;
    expectedIdf.forEach((value, i) => expect(transformer.idf[i]).toBeCloseTo(value, 12));
    tfidf.toDense().forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(waveB.text.tfidf_matrix[i][j], 12)));
    tfidf.toDense().forEach(row => expect(Math.sqrt(row.reduce((sum, value) => sum + value * value, 0))).toBeCloseTo(1, 12));
});

test('TfidfTransformer rejects ragged, negative, and non-finite counts', () => {
    const transformer = new TfidfTransformer();
    expect(() => transformer.fit([[1, 0], [1]])).toThrow('rectangular');
    expect(() => transformer.fit([[1, -1]])).toThrow('non-negative');
    expect(() => transformer.fit([[1, Infinity]])).toThrow('finite');
});

test('idf is unavailable when IDF weighting is disabled, matching sklearn', () => {
    const transformer = new TfidfTransformer({ useIdf: false });
    transformer.fit([[1, 0], [0, 1]]);
    expect(() => transformer.idf).toThrow('useIdf=false');
});

test('TfidfVectorizer survives serialization and feeds sparse text pipelines', () => {
    const vectorizer = new TfidfVectorizer({ ngramRange: [1, 2] });
    vectorizer.fit(documents);
    const revived = loadModel(JSON.stringify(vectorizer)) as TfidfVectorizer;
    expect(revived.getFeatureNamesOut()).toEqual(vectorizer.getFeatureNamesOut());
    expect(revived.transform(documents).toDense()).toEqual(vectorizer.transform(documents).toDense());

    const X = ['red apple sweet', 'green apple sweet', 'blue ocean deep', 'blue sea deep'];
    const y = [0, 0, 1, 1];
    const pipeline = new Pipeline({ steps: [
        ['tfidf', new TfidfVectorizer()],
        ['classifier', new MultinomialNB()],
    ] });
    pipeline.fit(X, y);
    expect(pipeline.predict(['sweet red apple', 'deep blue sea'])).toEqual([0, 1]);
});
