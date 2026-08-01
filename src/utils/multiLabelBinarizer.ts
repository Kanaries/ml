import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix } from '../data';

export type MultiLabel = string | number;
export interface MultiLabelBinarizerProps { classes?: MultiLabel[]; sparseOutput?: boolean; }
const compareLabels = (a: MultiLabel, b: MultiLabel): number => {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    const left = String(a), right = String(b); return left < right ? -1 : left > right ? 1 : 0;
};

/** Convert iterable sets of labels to a binary indicator matrix and back. */
export class MultiLabelBinarizer extends BaseEstimator {
    private classesParam?: MultiLabel[];
    private sparseOutput: boolean;
    private classesState: MultiLabel[] = [];
    private fitted = false;

    constructor(props: MultiLabelBinarizerProps = {}) {
        super();
        if (props.classes && new Set(props.classes.map(value => `${typeof value}:${String(value)}`)).size !== props.classes.length) throw new Error('classes must be unique');
        this.classesParam = props.classes?.slice();
        this.sparseOutput = props.sparseOutput ?? false;
    }
    public getParams(): Params { return { classes: this.classesParam?.slice(), sparseOutput: this.sparseOutput }; }
    public fit(y: MultiLabel[][]): void {
        if (!Array.isArray(y)) throw new Error('y must be an array of label collections');
        const found = this.classesParam ?? Array.from(new Set(y.flat()));
        this.classesState = this.classesParam ? found.slice() : found.slice().sort(compareLabels);
        this.fitted = true;
    }
    public transform(y: MultiLabel[][]): number[][] | CSRMatrix {
        if (!this.fitted) throw new Error('MultiLabelBinarizer is not fitted');
        const index = new Map(this.classesState.map((label, i) => [label, i]));
        const dense = y.map(labels => {
            const row = new Array(this.classesState.length).fill(0);
            for (const label of new Set(labels)) {
                const column = index.get(label);
                if (column === undefined) continue;
                row[column] = 1;
            }
            return row;
        });
        return this.sparseOutput ? CSRMatrix.fromDense(dense) : dense;
    }
    public fitTransform(y: MultiLabel[][]): number[][] | CSRMatrix { this.fit(y); return this.transform(y); }
    public inverseTransform(Y: number[][] | CSRMatrix): MultiLabel[][] {
        if (!this.fitted) throw new Error('MultiLabelBinarizer is not fitted');
        const dense = Y instanceof CSRMatrix ? Y.toDense() : Y;
        return dense.map(row => {
            if (row.length !== this.classesState.length || row.some(value => value !== 0 && value !== 1)) throw new Error('indicator matrix must be binary and match fitted classes');
            return row.flatMap((value, i) => value ? [this.classesState[i]] : []);
        });
    }
    public get classes(): MultiLabel[] { if (!this.fitted) throw new Error('MultiLabelBinarizer is not fitted'); return this.classesState.slice(); }
}

registerEstimator('MultiLabelBinarizer', MultiLabelBinarizer);
