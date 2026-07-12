import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import { hstack } from './featureUnion';

interface TransformerLike extends BaseEstimator {
    fit(X: number[][], y?: number[]): void;
    transform(X: number[][]): number[][];
}

export type ColumnTransformerEntry = [
    name: string,
    transformer: BaseEstimator | 'passthrough' | 'drop',
    columns: number[],
];

export interface ColumnTransformerProps {
    transformers: ColumnTransformerEntry[];
    /** What to do with columns not claimed by any transformer. Default 'drop'. */
    remainder?: 'drop' | 'passthrough';
}

function validateEntries(entries: ColumnTransformerEntry[]): void {
    if (!Array.isArray(entries) || entries.length === 0) {
        throw new Error('ColumnTransformer requires a non-empty transformers array');
    }
    const names = new Set<string>();
    for (const entry of entries) {
        if (!Array.isArray(entry) || entry.length !== 3 || typeof entry[0] !== 'string') {
            throw new Error('Each entry must be a [name, transformer, columns] triple');
        }
        const [name, t, columns] = entry;
        if (names.has(name)) throw new Error(`Duplicate transformer name "${name}"`);
        if (name.includes('__')) throw new Error(`Transformer name "${name}" must not contain "__"`);
        names.add(name);
        if (t !== 'passthrough' && t !== 'drop') {
            const cand = t as TransformerLike;
            if (!(t instanceof BaseEstimator) || typeof cand.fit !== 'function' || typeof cand.transform !== 'function') {
                throw new Error(`Transformer "${name}" must implement fit() and transform(), or be 'passthrough'/'drop'`);
            }
        }
        if (!Array.isArray(columns) || columns.some((c) => !Number.isInteger(c) || c < 0)) {
            throw new Error(`Columns for "${name}" must be non-negative integer indices`);
        }
    }
}

function selectColumns(X: number[][], columns: number[]): number[][] {
    return X.map((row) => columns.map((c) => {
        if (c >= row.length) throw new Error(`Column index ${c} out of range (row has ${row.length} features)`);
        return row[c];
    }));
}

/**
 * Applies different transformers to different column subsets and
 * concatenates the results (transformer outputs in declaration order,
 * remainder-passthrough columns last), like sklearn's ColumnTransformer.
 */
export class ColumnTransformer extends BaseEstimator {
    private transformers: ColumnTransformerEntry[];
    private remainder: 'drop' | 'passthrough';
    private remainderColumns: number[];
    private nFeaturesIn: number;
    private fitted: boolean;

    constructor(props: ColumnTransformerProps) {
        super();
        const { transformers, remainder = 'drop' } = props ?? {};
        validateEntries(transformers);
        if (remainder !== 'drop' && remainder !== 'passthrough') {
            throw new Error(`remainder must be 'drop' or 'passthrough', got "${remainder}"`);
        }
        this.transformers = transformers.map(([n, t, c]) => [n, t, c.slice()]);
        this.remainder = remainder;
        this.remainderColumns = [];
        this.nFeaturesIn = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            transformers: this.transformers.map(([n, t, c]) => [n, t, c.slice()]),
            remainder: this.remainder,
        };
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
            const est = this.getTransformer(name);
            if (typeof est === 'string') {
                throw new Error(`Cannot set params on '${est}' entry "${name}"`);
            }
            est.setParams(subParams);
        }
        return this;
    }

    public getTransformer(name: string): BaseEstimator | 'passthrough' | 'drop' {
        const entry = this.transformers.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown transformer "${name}". Names: ${this.transformers.map(([n]) => n).join(', ')}`);
        }
        return entry[1];
    }

    public fit(X: number[][], y?: number[]): void {
        if (X.length === 0) throw new Error('Cannot fit ColumnTransformer on empty X');
        const claimed = new Set<number>();
        for (const [, , columns] of this.transformers) {
            for (const c of columns) claimed.add(c);
        }
        this.nFeaturesIn = X[0].length;
        this.remainderColumns = [];
        for (let c = 0; c < X[0].length; c++) {
            if (!claimed.has(c)) this.remainderColumns.push(c);
        }
        for (const [, t, columns] of this.transformers) {
            if (t === 'passthrough' || t === 'drop') continue;
            (t as TransformerLike).fit(selectColumns(X, columns), y);
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) throw new Error('ColumnTransformer must be fitted before transform()');
        if (X.length > 0 && X[0].length !== this.nFeaturesIn) {
            throw new Error(`X has ${X[0].length} features, but ColumnTransformer was fitted with ${this.nFeaturesIn}`);
        }
        const blocks: number[][][] = [];
        for (const [, t, columns] of this.transformers) {
            if (t === 'drop') continue;
            const sub = selectColumns(X, columns);
            blocks.push(t === 'passthrough' ? sub : (t as TransformerLike).transform(sub));
        }
        if (this.remainder === 'passthrough' && this.remainderColumns.length > 0) {
            blocks.push(selectColumns(X, this.remainderColumns));
        }
        if (blocks.length === 0) {
            return X.map(() => []);
        }
        return hstack(blocks, X.length);
    }

    public fitTransform(X: number[][], y?: number[]): number[][] {
        this.fit(X, y);
        return this.transform(X);
    }
}
registerEstimator('ColumnTransformer', ColumnTransformer);
