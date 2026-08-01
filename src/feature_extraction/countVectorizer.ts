import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, TextDocuments } from '../data';

export interface CountVectorizerProps {
    lowercase?: boolean;
    stopWords?: string[];
    ngramRange?: [number, number];
    minDf?: number;
    maxDf?: number;
    maxFeatures?: number;
    binary?: boolean;
    vocabulary?: Record<string, number> | Map<string, number>;
}

function validateDocuments(documents: string[]): void {
    if (!Array.isArray(documents) || documents.some(document => typeof document !== 'string')) {
        throw new Error('CountVectorizer expects an array of string documents');
    }
}

export class CountVectorizer extends TransformerBase<TextDocuments, CSRMatrix> {
    public readonly acceptedInputKinds = ['text'] as const;
    private lowercase: boolean;
    private stopWords: string[];
    private ngramRange: [number, number];
    private minDf: number;
    private maxDf: number;
    private maxFeatures?: number;
    private binary: boolean;
    private fixedVocabulary?: Array<[string, number]>;
    private vocabularyState = new Map<string, number>();

    constructor(props: CountVectorizerProps = {}) {
        super();
        const {
            lowercase = true, stopWords = [], ngramRange = [1, 1], minDf = 1,
            maxDf = 1, maxFeatures, binary = false, vocabulary,
        } = props;
        if (typeof lowercase !== 'boolean' || typeof binary !== 'boolean' || !Array.isArray(stopWords) || stopWords.some(word => typeof word !== 'string')) throw new Error('lowercase and binary must be booleans and stopWords must contain strings');
        if (!Array.isArray(ngramRange) || !Number.isInteger(ngramRange[0]) || !Number.isInteger(ngramRange[1]) || ngramRange[0] < 1 || ngramRange[0] > ngramRange[1]) {
            throw new Error('ngramRange must contain positive integers [min, max]');
        }
        if (!Number.isFinite(minDf) || minDf <= 0 || !Number.isFinite(maxDf) || maxDf <= 0) throw new Error('minDf and maxDf must be positive finite numbers');
        if (maxFeatures !== undefined && (!Number.isInteger(maxFeatures) || maxFeatures < 1)) throw new Error('maxFeatures must be a positive integer');
        this.lowercase = lowercase;
        this.stopWords = stopWords.slice();
        this.ngramRange = [...ngramRange];
        this.minDf = minDf;
        this.maxDf = maxDf;
        this.maxFeatures = maxFeatures;
        this.binary = binary;
        if (vocabulary !== undefined) {
            const entries = vocabulary instanceof Map ? Array.from(vocabulary) : Object.entries(vocabulary);
            if (entries.some(([term]) => typeof term !== 'string')) throw new Error('vocabulary terms must be strings');
            const indices = entries.map(([, index]) => index).sort((a, b) => a - b);
            if (indices.some((index, i) => !Number.isInteger(index) || index !== i)) throw new Error('vocabulary indices must be contiguous from 0');
            this.fixedVocabulary = entries.map(([term, index]) => [term, index]);
            this.vocabularyState = new Map(this.fixedVocabulary);
        }
    }

    public getParams(): Params {
        return {
            lowercase: this.lowercase, stopWords: this.stopWords.slice(), ngramRange: [...this.ngramRange],
            minDf: this.minDf, maxDf: this.maxDf, maxFeatures: this.maxFeatures, binary: this.binary,
            vocabulary: this.fixedVocabulary === undefined ? undefined : new Map(this.fixedVocabulary),
        };
    }

    private analyze(document: string): string[] {
        const normalized = this.lowercase ? document.toLowerCase() : document;
        const stop = new Set(this.stopWords.map(word => this.lowercase ? word.toLowerCase() : word));
        const unigrams = normalized.match(/[\p{L}\p{N}_]{2,}/gu)?.filter(token => !stop.has(token)) ?? [];
        const output: string[] = [];
        for (let size = this.ngramRange[0]; size <= this.ngramRange[1]; size++) {
            for (let start = 0; start + size <= unigrams.length; start++) output.push(unigrams.slice(start, start + size).join(' '));
        }
        return output;
    }

    public fit(documents: string[]): void {
        validateDocuments(documents);
        if (this.fixedVocabulary) {
            this.vocabularyState = new Map(this.fixedVocabulary);
            return;
        }
        const termFrequency = new Map<string, number>();
        const documentFrequency = new Map<string, number>();
        for (const document of documents) {
            const tokens = this.analyze(document);
            for (const token of tokens) termFrequency.set(token, (termFrequency.get(token) ?? 0) + 1);
            for (const token of new Set(tokens)) documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
        }
        const minDocuments = this.minDf < 1 ? Math.ceil(this.minDf * documents.length) : Math.floor(this.minDf);
        const maxDocuments = this.maxDf <= 1 ? Math.floor(this.maxDf * documents.length) : Math.floor(this.maxDf);
        if (maxDocuments < minDocuments) throw new Error('maxDf corresponds to fewer documents than minDf');
        let terms = Array.from(termFrequency.keys()).filter(term => {
            const frequency = documentFrequency.get(term)!;
            return frequency >= minDocuments && frequency <= maxDocuments;
        });
        if (this.maxFeatures !== undefined && terms.length > this.maxFeatures) {
            terms.sort((a, b) => termFrequency.get(b)! - termFrequency.get(a)! || (a < b ? -1 : a > b ? 1 : 0));
            terms = terms.slice(0, this.maxFeatures);
        }
        terms.sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
        if (terms.length === 0) throw new Error('empty vocabulary; perhaps the documents only contain stop words');
        this.vocabularyState = new Map(terms.map((term, index) => [term, index]));
    }

    public transform(documents: string[]): CSRMatrix {
        validateDocuments(documents);
        if (this.vocabularyState.size === 0) throw new Error('CountVectorizer is not fitted');
        const data: number[] = [], indices: number[] = [], indptr = [0];
        for (const document of documents) {
            const counts = new Map<number, number>();
            for (const token of this.analyze(document)) {
                const index = this.vocabularyState.get(token);
                if (index !== undefined) counts.set(index, this.binary ? 1 : (counts.get(index) ?? 0) + 1);
            }
            for (const [index, value] of Array.from(counts).sort((a, b) => a[0] - b[0])) {
                indices.push(index); data.push(value);
            }
            indptr.push(data.length);
        }
        return new CSRMatrix(data, indices, indptr, [documents.length, this.vocabularyState.size]);
    }

    public get vocabulary(): ReadonlyMap<string, number> { if (this.vocabularyState.size === 0) throw new Error('CountVectorizer is not fitted'); return new Map(this.vocabularyState); }
    public getFeatureNamesOut(): string[] {
        if (this.vocabularyState.size === 0) throw new Error('CountVectorizer is not fitted');
        return Array.from(this.vocabularyState).sort((a, b) => a[1] - b[1]).map(([term]) => term);
    }
}
registerEstimator('CountVectorizer', CountVectorizer);
