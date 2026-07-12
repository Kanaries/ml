import { BaseEstimator, Params, registerEstimator } from '../base/estimator';

interface TransformerLike extends BaseEstimator {
    fit(X: number[][], y?: number[]): void;
    transform(X: number[][]): number[][];
}

export interface FeatureUnionProps {
    /** [name, transformer] pairs whose outputs are concatenated column-wise. */
    transformerList: Array<[name: string, transformer: BaseEstimator]>;
}

function validateList(list: FeatureUnionProps['transformerList']): void {
    if (!Array.isArray(list) || list.length === 0) {
        throw new Error('FeatureUnion requires a non-empty transformerList');
    }
    const names = new Set<string>();
    for (const entry of list) {
        if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string') {
            throw new Error('Each entry must be a [name, transformer] pair');
        }
        const [name, t] = entry;
        if (names.has(name)) throw new Error(`Duplicate transformer name "${name}"`);
        if (name.includes('__')) throw new Error(`Transformer name "${name}" must not contain "__"`);
        names.add(name);
        const cand = t as TransformerLike;
        if (!(t instanceof BaseEstimator) || typeof cand.fit !== 'function' || typeof cand.transform !== 'function') {
            throw new Error(`Transformer "${name}" must implement fit() and transform()`);
        }
    }
}

/** Concatenates the outputs of several transformers, like sklearn's FeatureUnion. */
export class FeatureUnion extends BaseEstimator {
    private transformerList: Array<[string, BaseEstimator]>;

    constructor(props: FeatureUnionProps) {
        super();
        const { transformerList } = props ?? {};
        validateList(transformerList);
        this.transformerList = transformerList.map(([name, t]) => [name, t]);
    }

    public getParams(): Params {
        return { transformerList: this.transformerList.map(([name, t]) => [name, t]) };
    }

    /** Supports nested `name__param` addressing like Pipeline. */
    public setParams(params: Params): this {
        const own: Params = {};
        const nested = new Map<string, Params>();
        for (const key of Object.keys(params)) {
            const idx = key.indexOf('__');
            if (idx > 0) {
                const name = key.slice(0, idx);
                if (!nested.has(name)) nested.set(name, {});
                nested.get(name)![key.slice(idx + 2)] = params[key];
            } else {
                own[key] = params[key];
            }
        }
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [name, subParams] of nested) {
            this.getTransformer(name).setParams(subParams);
        }
        return this;
    }

    public getTransformer(name: string): BaseEstimator {
        const entry = this.transformerList.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown transformer "${name}". Names: ${this.transformerList.map(([n]) => n).join(', ')}`);
        }
        return entry[1];
    }

    public fit(X: number[][], y?: number[]): void {
        for (const [, t] of this.transformerList) {
            (t as TransformerLike).fit(X, y);
        }
    }

    public transform(X: number[][]): number[][] {
        const blocks = this.transformerList.map(([, t]) => (t as TransformerLike).transform(X));
        return hstack(blocks, X.length);
    }

    public fitTransform(X: number[][], y?: number[]): number[][] {
        this.fit(X, y);
        return this.transform(X);
    }
}
registerEstimator('FeatureUnion', FeatureUnion);

export function hstack(blocks: number[][][], nRows: number): number[][] {
    for (const block of blocks) {
        if (block.length !== nRows) {
            throw new Error(`Transformer output has ${block.length} rows, expected ${nRows}`);
        }
    }
    const out: number[][] = [];
    for (let i = 0; i < nRows; i++) {
        const row: number[] = [];
        for (const block of blocks) {
            for (const v of block[i]) row.push(v);
        }
        out.push(row);
    }
    return out;
}
