import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import type { FeatureData, FeatureDataKind, NumericMatrix } from '../data';
import { CSRMatrix, isCSRMatrix } from '../data';

/**
 * Structural interfaces: pipeline steps are validated at runtime by shape,
 * so any estimator following the contract works, including ones from other
 * modules loaded lazily.
 */
interface TransformerLike extends BaseEstimator {
    readonly acceptedInputKinds?: readonly FeatureDataKind[];
    fit(X: FeatureData, y?: number[]): void;
    transform(X: FeatureData): FeatureData;
}

function featureDataKind(X: FeatureData): FeatureDataKind {
    if (isCSRMatrix(X)) return 'csr';
    return X.length > 0 && typeof X[0] === 'string' ? 'text' : 'dense';
}

function transformAtStep(name: string, transformer: TransformerLike, X: FeatureData): FeatureData {
    const kind = featureDataKind(X);
    const accepted = transformer.acceptedInputKinds ?? ['dense'];
    if (!accepted.includes(kind)) {
        throw new Error(`Pipeline step "${name}" does not accept ${kind} input`);
    }
    try {
        return transformer.transform(X);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Pipeline step "${name}" failed to transform ${kind} input: ${message}`);
    }
}

function fitTransformerAtStep(name: string, transformer: TransformerLike, X: FeatureData, y?: number[]): void {
    const kind = featureDataKind(X);
    const accepted = transformer.acceptedInputKinds ?? ['dense'];
    if (!accepted.includes(kind)) {
        throw new Error(`Pipeline step "${name}" does not accept ${kind} input`);
    }
    transformer.fit(X, y);
}
interface PredictorLike extends BaseEstimator {
    readonly acceptedInputKinds?: readonly FeatureDataKind[];
    fit(X: NumericMatrix, y?: number[], sampleWeight?: number[]): void;
    predict(X: NumericMatrix): number[];
    predictProba?(X: NumericMatrix): number[][];
    score?(X: NumericMatrix, y: number[]): number;
}

export type PipelineStep = readonly [name: string, estimator: BaseEstimator];

type Last<T extends readonly unknown[]> = T extends readonly [...unknown[], infer L]
    ? L
    : T extends readonly (infer E)[] ? E : never;
type StepEstimator<T> = T extends readonly [string, infer E] ? E : never;
type TransformerOutput<T> = T extends { transform(X: never): infer O }
    ? Extract<O, FeatureData>
    : FeatureData;

/** Output type of the final step when it is a transformer. */
export type PipelineOutput<TSteps extends readonly PipelineStep[]> =
    TransformerOutput<StepEstimator<Last<TSteps>>>;

export interface PipelineProps<TSteps extends readonly PipelineStep[] = readonly PipelineStep[]> {
    /** Ordered [name, estimator] pairs; all but the last must be transformers. */
    steps: TSteps;
}

function isTransformer(est: unknown): est is TransformerLike {
    return est instanceof BaseEstimator
        && typeof (est as TransformerLike).fit === 'function'
        && typeof (est as TransformerLike).transform === 'function';
}

function requireNumericData(X: FeatureData, consumer: string): NumericMatrix {
    if (isCSRMatrix(X)) return X;
    if (X.length === 0 || Array.isArray(X[0])) return X as number[][];
    throw new Error(
        `${consumer} received raw string documents. Add a text transformer before the numeric estimator.`,
    );
}

function numericAtFinalStep(name: string, predictor: PredictorLike, X: FeatureData): NumericMatrix {
    const numeric = requireNumericData(X, `Pipeline step "${name}"`);
    const kind = featureDataKind(numeric);
    const accepted = predictor.acceptedInputKinds ?? ['dense'];
    if (!accepted.includes(kind)) {
        throw new Error(`Pipeline step "${name}" does not accept ${kind} input`);
    }
    return numeric;
}

function validateSteps(steps: readonly PipelineStep[]): void {
    if (!Array.isArray(steps) || steps.length === 0) {
        throw new Error('Pipeline requires a non-empty steps array');
    }
    const names = new Set<string>();
    for (const step of steps) {
        if (!Array.isArray(step) || step.length !== 2 || typeof step[0] !== 'string') {
            throw new Error('Each pipeline step must be a [name, estimator] pair');
        }
        const [name, est] = step;
        if (names.has(name)) throw new Error(`Duplicate step name "${name}"`);
        if (name.includes('__')) throw new Error(`Step name "${name}" must not contain "__"`);
        names.add(name);
        if (!(est instanceof BaseEstimator)) {
            throw new Error(`Step "${name}" is not an estimator`);
        }
    }
    for (const [name, est] of steps.slice(0, -1)) {
        if (!isTransformer(est)) {
            throw new Error(`Intermediate step "${name}" must implement fit() and transform()`);
        }
    }
}

/**
 * Chain of transformers with a final estimator, mirroring sklearn's
 * `Pipeline`. Nested params are addressable as `step__param` in `setParams`
 * (and therefore in grid search): `pipe.setParams({ svc__C: 10 })`.
 */
export class Pipeline<const TSteps extends readonly PipelineStep[] = readonly PipelineStep[]> extends BaseEstimator {
    private steps: PipelineStep[];

    constructor(props: PipelineProps<TSteps>) {
        super();
        const { steps } = props ?? {};
        validateSteps(steps);
        this.steps = steps.map(([name, est]) => [name, est]);
    }

    public getParams(): Params {
        return { steps: this.steps.map(([name, est]) => [name, est]) };
    }

    /** Supports both own params and nested `step__param` addressing. */
    public setParams(params: Params): this {
        const own: Params = {};
        const nested = new Map<string, Params>();
        for (const key of Object.keys(params)) {
            const idx = key.indexOf('__');
            if (idx > 0) {
                const stepName = key.slice(0, idx);
                const subKey = key.slice(idx + 2);
                if (!nested.has(stepName)) nested.set(stepName, {});
                nested.get(stepName)![subKey] = params[key];
            } else {
                own[key] = params[key];
            }
        }
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [stepName, subParams] of nested) {
            this.getStep(stepName).setParams(subParams);
        }
        return this;
    }

    public getStep(name: string): BaseEstimator {
        const step = this.steps.find(([n]) => n === name);
        if (!step) {
            throw new Error(`Unknown step "${name}". Steps: ${this.steps.map(([n]) => n).join(', ')}`);
        }
        return step[1];
    }

    public get namedSteps(): Record<string, BaseEstimator> {
        return Object.fromEntries(this.steps);
    }

    private get finalStep(): BaseEstimator {
        return this.steps[this.steps.length - 1][1];
    }

    /** Fit all transformers, transforming the data through, then fit the final estimator. */
    public fit(X: FeatureData, y?: number[], sampleWeight?: number[]): void {
        const Xt = this.fitIntermediate(X, y);
        const finalStep = this.finalStep;
        const name = this.steps[this.steps.length - 1][0];
        if (isTransformer(finalStep)) {
            fitTransformerAtStep(name, finalStep, Xt, y);
            return;
        }
        const final = finalStep as PredictorLike;
        final.fit(numericAtFinalStep(name, final, Xt), y, sampleWeight);
    }

    private fitIntermediate(X: FeatureData, y?: number[]): FeatureData {
        let Xt = X;
        for (const [name, est] of this.steps.slice(0, -1)) {
            const t = est as TransformerLike;
            fitTransformerAtStep(name, t, Xt, y);
            Xt = transformAtStep(name, t, Xt);
        }
        return Xt;
    }

    private applyIntermediate(X: FeatureData): FeatureData {
        let Xt = X;
        for (const [name, est] of this.steps.slice(0, -1)) {
            Xt = transformAtStep(name, est as TransformerLike, Xt);
        }
        return Xt;
    }

    public predict(X: FeatureData): number[] {
        const final = this.finalStep as PredictorLike;
        if (typeof final.predict !== 'function') {
            throw new Error('The final pipeline step does not implement predict()');
        }
        const name = this.steps[this.steps.length - 1][0];
        return final.predict(numericAtFinalStep(name, final, this.applyIntermediate(X)));
    }

    public predictProba(X: FeatureData): number[][] {
        const final = this.finalStep as PredictorLike;
        if (typeof final.predictProba !== 'function') {
            throw new Error('The final pipeline step does not implement predictProba()');
        }
        const name = this.steps[this.steps.length - 1][0];
        return final.predictProba(numericAtFinalStep(name, final, this.applyIntermediate(X)));
    }

    /** Transform through every step (requires the final step to be a transformer too). */
    public transform(X: FeatureData): PipelineOutput<TSteps> {
        const final = this.finalStep;
        if (!isTransformer(final)) {
            throw new Error('The final pipeline step does not implement transform()');
        }
        return transformAtStep(this.steps[this.steps.length - 1][0], final, this.applyIntermediate(X)) as PipelineOutput<TSteps>;
    }

    public fitTransform(X: FeatureData, y?: number[]): PipelineOutput<TSteps> {
        const final = this.finalStep;
        if (!isTransformer(final)) {
            throw new Error('The final pipeline step does not implement transform()');
        }
        const Xt = this.fitIntermediate(X, y);
        const name = this.steps[this.steps.length - 1][0];
        fitTransformerAtStep(name, final, Xt, y);
        return transformAtStep(name, final, Xt) as PipelineOutput<TSteps>;
    }

    public score(X: FeatureData, y: number[]): number {
        const final = this.finalStep as PredictorLike;
        if (typeof final.score !== 'function') {
            throw new Error('The final pipeline step does not implement score()');
        }
        const name = this.steps[this.steps.length - 1][0];
        return final.score(numericAtFinalStep(name, final, this.applyIntermediate(X)), y);
    }
}
registerEstimator('Pipeline', Pipeline);

/** `makePipeline(new StandardScaler(), new SVC())` — names derived from class names. */
type NamedSteps<T extends readonly BaseEstimator[]> = {
    readonly [K in keyof T]: readonly [name: string, estimator: T[K]];
};

export function makePipeline<const T extends readonly BaseEstimator[]>(...estimators: T): Pipeline<NamedSteps<T>> {
    const counts = new Map<string, number>();
    const steps: PipelineStep[] = estimators.map((est) => {
        const base = est.constructor.name.toLowerCase();
        const seen = counts.get(base) ?? 0;
        counts.set(base, seen + 1);
        return [seen === 0 ? base : `${base}-${seen + 1}`, est];
    });
    // if a later duplicate forced numbering, renumber the first occurrence too
    for (const [base, total] of counts) {
        if (total > 1) {
            const firstIndex = steps.findIndex(([name]) => name === base);
            if (firstIndex >= 0) steps[firstIndex] = [`${base}-1`, steps[firstIndex][1]];
        }
    }
    return new Pipeline({ steps }) as Pipeline<NamedSteps<T>>;
}
