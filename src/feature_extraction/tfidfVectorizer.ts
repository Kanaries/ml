import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, TextDocuments } from '../data';
import { CountVectorizer, CountVectorizerProps } from './countVectorizer';
import { TfidfTransformer, TfidfTransformerProps } from './tfidfTransformer';

export interface TfidfVectorizerProps extends CountVectorizerProps, TfidfTransformerProps {}

export class TfidfVectorizer extends TransformerBase<TextDocuments, CSRMatrix> {
    public readonly acceptedInputKinds = ['text'] as const;
    private props: TfidfVectorizerProps;
    private countState?: CountVectorizer;
    private tfidfState?: TfidfTransformer;
    constructor(props: TfidfVectorizerProps = {}) {
        super();
        const countParams = new CountVectorizer(props).getParams();
        const tfidfParams = new TfidfTransformer(props).getParams();
        this.props = { ...countParams, ...tfidfParams } as TfidfVectorizerProps;
    }
    public getParams(): Params {
        return {
            ...this.props,
            stopWords: this.props.stopWords?.slice(),
            ngramRange: this.props.ngramRange === undefined ? undefined : [...this.props.ngramRange],
            vocabulary: this.props.vocabulary instanceof Map ? new Map(this.props.vocabulary) : this.props.vocabulary === undefined ? undefined : { ...this.props.vocabulary },
        };
    }
    public fit(documents: string[]): void {
        const { norm, useIdf, smoothIdf, sublinearTf, ...countProps } = this.props;
        this.countState = new CountVectorizer(countProps);
        const counts = this.countState.fitTransform(documents);
        this.tfidfState = new TfidfTransformer({ norm, useIdf, smoothIdf, sublinearTf });
        this.tfidfState.fit(counts);
    }
    public transform(documents: string[]): CSRMatrix {
        if (!this.countState || !this.tfidfState) throw new Error('TfidfVectorizer is not fitted');
        return this.tfidfState.transform(this.countState.transform(documents));
    }
    public get vocabulary(): ReadonlyMap<string, number> {
        if (!this.countState) throw new Error('TfidfVectorizer is not fitted');
        return this.countState.vocabulary;
    }
    public get idf(): number[] {
        if (!this.tfidfState) throw new Error('TfidfVectorizer is not fitted');
        return this.tfidfState.idf;
    }
    public getFeatureNamesOut(): string[] {
        if (!this.countState) throw new Error('TfidfVectorizer is not fitted');
        return this.countState.getFeatureNamesOut();
    }
}
registerEstimator('TfidfVectorizer', TfidfVectorizer);
