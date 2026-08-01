import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, TextDocuments } from '../data';
import { hashedRows } from './hashing';

export interface HashingVectorizerProps { nFeatures?: number; lowercase?: boolean; stopWords?: string[]; ngramRange?: [number, number]; binary?: boolean; norm?: 'l1' | 'l2' | null; alternateSign?: boolean; }

export class HashingVectorizer extends TransformerBase<TextDocuments, CSRMatrix> {
    public readonly acceptedInputKinds = ['text'] as const;
    private nFeatures: number; private lowercase: boolean; private stopWords: string[]; private ngramRange: [number, number]; private binary: boolean; private norm: 'l1' | 'l2' | null; private alternateSign: boolean;
    constructor(props: HashingVectorizerProps = {}) {
        super(); const { nFeatures = 2 ** 20, lowercase = true, stopWords = [], ngramRange = [1, 1], binary = false, norm = 'l2', alternateSign = true } = props;
        if (!Number.isInteger(nFeatures) || nFeatures < 1 || !Array.isArray(ngramRange) || !Number.isInteger(ngramRange[0]) || !Number.isInteger(ngramRange[1]) || ngramRange[0] < 1 || ngramRange[0] > ngramRange[1] || !['l1', 'l2', null].includes(norm)) throw new Error('invalid HashingVectorizer parameters');
        this.nFeatures = nFeatures; this.lowercase = lowercase; this.stopWords = stopWords.slice(); this.ngramRange = [...ngramRange]; this.binary = binary; this.norm = norm; this.alternateSign = alternateSign;
    }
    public getParams(): Params { return { nFeatures: this.nFeatures, lowercase: this.lowercase, stopWords: this.stopWords.slice(), ngramRange: [...this.ngramRange], binary: this.binary, norm: this.norm, alternateSign: this.alternateSign }; }
    private analyze(document: string): string[] {
        const normalized = this.lowercase ? document.toLowerCase() : document, stop = new Set(this.stopWords.map(word => this.lowercase ? word.toLowerCase() : word));
        const tokens = normalized.match(/[\p{L}\p{N}_]{2,}/gu)?.filter(token => !stop.has(token)) ?? [], output: string[] = [];
        for (let size = this.ngramRange[0]; size <= this.ngramRange[1]; size++) for (let start = 0; start + size <= tokens.length; start++) output.push(tokens.slice(start, start + size).join(' '));
        return output;
    }
    public fit(documents: string[]): void { if (!Array.isArray(documents) || documents.some(value => typeof value !== 'string')) throw new Error('HashingVectorizer expects string documents'); }
    public transform(documents: string[]): CSRMatrix {
        if (!Array.isArray(documents) || documents.some(value => typeof value !== 'string')) throw new Error('HashingVectorizer expects string documents');
        const matrix = hashedRows(documents.map(document => this.analyze(document).map(token => [token, 1] as [string, number])), this.nFeatures, this.alternateSign);
        const data = Array.from(matrix.data), indices = Array.from(matrix.indices), indptr = Array.from(matrix.indptr);
        for (let row = 0; row < documents.length; row++) {
            const start = indptr[row], end = indptr[row + 1];
            if (this.binary) for (let position = start; position < end; position++) data[position] = 1;
            let scale = 1;
            if (this.norm === 'l1') { scale = 0; for (let position = start; position < end; position++) scale += Math.abs(data[position]); }
            else if (this.norm === 'l2') { scale = 0; for (let position = start; position < end; position++) scale += data[position] ** 2; scale = Math.sqrt(scale); }
            if (scale !== 0 && scale !== 1) for (let position = start; position < end; position++) data[position] /= scale;
        }
        return new CSRMatrix(data, indices, indptr, matrix.shape);
    }
}
registerEstimator('HashingVectorizer', HashingVectorizer);
