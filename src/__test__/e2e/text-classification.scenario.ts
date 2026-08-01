import { freezeScenario } from './scenario';

export const textClassification = freezeScenario({
    id: 'text-classification', title: 'Sparse text classification', frozenAt: '2026-07-31',
    dataset: { name: '20 Newsgroups subset', source: 'sklearn.datasets.fetch_20newsgroups', protocol: 'four fixed categories, train/test split supplied by dataset' },
    workflow: ['TfidfVectorizer', 'MultinomialNB', 'Pipeline', 'prediction and macro-F1 parity'],
    algorithms: { include: ['TfidfVectorizer', 'MultinomialNB', 'Pipeline'], exclude: [] },
    parity: { state: 'pending', blockedBy: ['Wave B: TfidfVectorizer'], reason: 'Text vectorizers are intentionally scheduled for Wave B.' },
});
