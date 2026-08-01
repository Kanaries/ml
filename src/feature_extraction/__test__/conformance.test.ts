import { getRegisteredEstimators, loadModel } from '../../base';
import { CountVectorizer } from '../countVectorizer';
import { TfidfTransformer } from '../tfidfTransformer';
import { TfidfVectorizer } from '../tfidfVectorizer';

const documents = ['red apple sweet', 'green apple sweet', 'blue ocean deep', 'blue sea deep'];

describe('text transformer conformance', () => {
    test('all vectorizers are registered', () => {
        for (const name of ['CountVectorizer', 'TfidfTransformer', 'TfidfVectorizer']) {
            expect(getRegisteredEstimators().has(name)).toBe(true);
        }
    });

    test('constructor inputs and getParams are defensive and clone is unfitted', () => {
        const stopWords = ['and'];
        const source = new TfidfVectorizer({ stopWords, ngramRange: [1, 2], sublinearTf: true });
        const cloned = source.clone() as TfidfVectorizer;
        expect(cloned).not.toBe(source);
        expect(cloned.getParams()).toEqual(source.getParams());
        stopWords.push('document');
        expect(cloned.getParams().stopWords).toEqual(['and']);
        expect(() => cloned.transform(documents)).toThrow('not fitted');
    });

    test('fit does not mutate text input and repeated fits are deterministic', () => {
        const input = documents.slice();
        const vectorizer = new CountVectorizer({ ngramRange: [1, 2] });
        const first = vectorizer.fitTransform(input).toDense();
        const second = vectorizer.fitTransform(input).toDense();
        expect(input).toEqual(documents);
        expect(second).toEqual(first);
    });

    test('setParams clears fitted state through the base estimator contract', () => {
        const vectorizer = new CountVectorizer();
        vectorizer.fit(documents);
        vectorizer.setParams({ binary: true });
        expect(() => vectorizer.transform(documents)).toThrow('not fitted');
    });

    test('all fitted vectorizers survive serialization with output parity', () => {
        const count = new CountVectorizer({ binary: true });
        const counts = count.fitTransform(documents);
        const countRevived = loadModel(JSON.stringify(count)) as CountVectorizer;
        expect(countRevived.transform(documents).toDense()).toEqual(counts.toDense());

        const tfidf = new TfidfTransformer({ norm: 'l1', sublinearTf: true });
        const expected = tfidf.fitTransform(counts).toDense();
        const tfidfRevived = loadModel(JSON.stringify(tfidf)) as TfidfTransformer;
        expect(tfidfRevived.transform(counts).toDense()).toEqual(expected);

        const combined = new TfidfVectorizer({ ngramRange: [1, 2] });
        const combinedExpected = combined.fitTransform(documents).toDense();
        const combinedRevived = loadModel(JSON.stringify(combined)) as TfidfVectorizer;
        expect(combinedRevived.transform(documents).toDense()).toEqual(combinedExpected);
    });
});
