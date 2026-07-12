import { BaseEstimator, Params, registerEstimator } from '../base/estimator';

/**
 * Structural interfaces: pipeline steps are validated at runtime by shape,
 * so any estimator following the contract works, including ones from other
 * modules loaded lazily.
 */
interface TransformerLike extends BaseEstimator {
    fit(X: number[][], y?: number[]): void;
    transform(X: number[][]): number[][];
}
interface PredictorLike extends BaseEstimator {
    fit(X: number[][], y?: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
    predictProba?(X: number[][]): number[][];
    score?(X: number[][], y: number[]): number;
}

export type PipelineStep = [name: string, estimator: BaseEstimator];

export interface PipelineProps {
    /** Ordered [name, estimator] pairs; all but the last must be transformers. */
    steps: PipelineStep[];
}

function isTransformer(est: unknown): est is TransformerLike {
    return est instanceof BaseEstimator
        && typeof (est as TransformerLike).fit === 'function'
        && typeof (est as TransformerLike).transform === 'function';
}

function validateSteps(steps: PipelineStep[]): void {
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
export class Pipeline extends BaseEstimator {
    private steps: PipelineStep[];

    constructor(props: PipelineProps) {
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
    public fit(X: number[][], y?: number[], sampleWeight?: number[]): void {
        const Xt = this.fitIntermediate(X, y);
        (this.finalStep as PredictorLike).fit(Xt, y, sampleWeight);
    }

    private fitIntermediate(X: number[][], y?: number[]): number[][] {
        let Xt = X;
        for (const [, est] of this.steps.slice(0, -1)) {
            const t = est as TransformerLike;
            t.fit(Xt, y);
            Xt = t.transform(Xt);
        }
        return Xt;
    }

    private applyIntermediate(X: number[][]): number[][] {
        let Xt = X;
        for (const [, est] of this.steps.slice(0, -1)) {
            Xt = (est as TransformerLike).transform(Xt);
        }
        return Xt;
    }

    public predict(X: number[][]): number[] {
        const final = this.finalStep as PredictorLike;
        if (typeof final.predict !== 'function') {
            throw new Error('The final pipeline step does not implement predict()');
        }
        return final.predict(this.applyIntermediate(X));
    }

    public predictProba(X: number[][]): number[][] {
        const final = this.finalStep as PredictorLike;
        if (typeof final.predictProba !== 'function') {
            throw new Error('The final pipeline step does not implement predictProba()');
        }
        return final.predictProba(this.applyIntermediate(X));
    }

    /** Transform through every step (requires the final step to be a transformer too). */
    public transform(X: number[][]): number[][] {
        const final = this.finalStep;
        if (!isTransformer(final)) {
            throw new Error('The final pipeline step does not implement transform()');
        }
        return final.transform(this.applyIntermediate(X));
    }

    public fitTransform(X: number[][], y?: number[]): number[][] {
        const final = this.finalStep;
        if (!isTransformer(final)) {
            throw new Error('The final pipeline step does not implement transform()');
        }
        const Xt = this.fitIntermediate(X, y);
        final.fit(Xt, y);
        return final.transform(Xt);
    }

    public score(X: number[][], y: number[]): number {
        const final = this.finalStep as PredictorLike;
        if (typeof final.score !== 'function') {
            throw new Error('The final pipeline step does not implement score()');
        }
        return final.score(this.applyIntermediate(X), y);
    }
}
registerEstimator('Pipeline', Pipeline);

/** `makePipeline(new StandardScaler(), new SVC())` — names derived from class names. */
export function makePipeline(...estimators: BaseEstimator[]): Pipeline {
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
            const first = steps.find(([name]) => name === base);
            if (first) first[0] = `${base}-1`;
        }
    }
    return new Pipeline({ steps });
}
