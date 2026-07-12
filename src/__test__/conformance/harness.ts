/**
 * Generic conformance checks that every estimator in the library must pass.
 *
 * Each module owns a `__test__/conformance.test.ts` that calls
 * `runEstimatorConformance` with one spec per estimator. The harness verifies
 * the BaseEstimator contract: registration, getParams/setParams, clone,
 * deterministic refit, and serialize → revive → identical-behavior parity.
 */
import { BaseEstimator, getRegisteredEstimators, loadModel } from '../../base/estimator';
import {
    Dataset,
    binaryDataset,
    binaryFeaturesDataset,
    blobsDataset,
    countsDataset,
    multiclassDataset,
    regressionDataset,
} from './datasets';

export type EstimatorKind =
    /** fit(X, y) + predict(X) */
    | 'classifier'
    | 'regressor'
    /** fitPredict(X) */
    | 'cluster'
    /** fit(X[, y]) + transform(X) */
    | 'transformer'
    /** fitTransform(X) only (no out-of-sample transform) */
    | 'embedding'
    /** fit(X) + predict(X) */
    | 'outlier';

export type DatasetName = 'binary' | 'multiclass' | 'regression' | 'blobs' | 'counts' | 'binaryFeatures';

export interface EstimatorSpec {
    /** Name used in registerEstimator(); also the describe-block label. */
    name: string;
    /** A fully-seeded, small-config instance. Must be deterministic. */
    create: () => BaseEstimator;
    kind: EstimatorKind;
    /** Which dataset to fit on; defaults per kind. */
    dataset?: DatasetName;
    /**
     * Set when refitting a clone cannot reproduce identical output (estimator
     * has irreducible randomness). The serialization parity check still runs.
     */
    nonDeterministic?: boolean;
}

const DATASETS: Record<DatasetName, () => Dataset> = {
    binary: binaryDataset,
    multiclass: multiclassDataset,
    regression: regressionDataset,
    blobs: blobsDataset,
    counts: countsDataset,
    binaryFeatures: binaryFeaturesDataset,
};

const DEFAULT_DATASET: Record<EstimatorKind, DatasetName> = {
    classifier: 'multiclass',
    regressor: 'regression',
    cluster: 'blobs',
    transformer: 'blobs',
    embedding: 'blobs',
    outlier: 'blobs',
};

interface FittableEstimator extends BaseEstimator {
    fit?(X: number[][], y?: number[]): void;
    predict?(X: number[][]): number[];
    transform?(X: number[][]): number[][];
    fitTransform?(X: number[][], y?: number[]): number[][];
    fitPredict?(X: number[][]): number[];
}

/** Fit the estimator on the dataset and return its primary output. */
function fitAndRun(est: BaseEstimator, kind: EstimatorKind, data: Dataset): unknown {
    const e = est as FittableEstimator;
    switch (kind) {
        case 'classifier':
        case 'regressor':
            e.fit!(data.X, data.y);
            return e.predict!(data.X);
        case 'cluster':
            return e.fitPredict!(data.X);
        case 'transformer':
            e.fit!(data.X, data.y);
            return e.transform!(data.X);
        case 'embedding':
            return e.fitTransform!(data.X, data.y);
        case 'outlier':
            e.fit!(data.X);
            return e.predict!(data.X);
    }
}

/** Re-run the estimator's inference path without refitting (where possible). */
function runFitted(est: BaseEstimator, kind: EstimatorKind, data: Dataset): unknown {
    const e = est as FittableEstimator;
    switch (kind) {
        case 'classifier':
        case 'regressor':
        case 'outlier':
            return e.predict!(data.X);
        case 'transformer':
            return e.transform!(data.X);
        case 'cluster':
        case 'embedding':
            return null; // no out-of-sample inference path
    }
}

export function runEstimatorConformance(specs: EstimatorSpec[]): void {
    for (const spec of specs) {
        const data = DATASETS[spec.dataset ?? DEFAULT_DATASET[spec.kind]]();
        describe(`${spec.name} conformance`, () => {
            it('is registered under its declared name', () => {
                const est = spec.create();
                expect(getRegisteredEstimators().get(spec.name)).toBe(est.constructor);
            });

            it('getParams/setParams round-trips and validates keys', () => {
                const est = spec.create();
                const params = est.getParams();
                est.setParams(params);
                expect(est.getParams()).toEqual(params);
                expect(() => est.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
            });

            it('clone copies params onto a fresh instance', () => {
                const est = spec.create();
                const copy = est.clone();
                expect(copy).not.toBe(est);
                expect(copy.constructor).toBe(est.constructor);
                expect(copy.getParams()).toEqual(est.getParams());
            });

            it('fits and produces output on its reference dataset', () => {
                const est = spec.create();
                const out = fitAndRun(est, spec.kind, data);
                expect(out).toBeTruthy();
                expect((out as unknown[]).length).toBe(data.X.length);
            });

            (spec.nonDeterministic ? it.skip : it)('refitting a clone reproduces identical output', () => {
                const est = spec.create();
                const out1 = fitAndRun(est, spec.kind, data);
                const out2 = fitAndRun(est.clone(), spec.kind, data);
                expect(out2).toEqual(out1);
            });

            it('survives serialize → JSON text → revive with identical state and behavior', () => {
                const est = spec.create();
                const out1 = fitAndRun(est, spec.kind, data);
                const revived = loadModel(JSON.stringify(est));
                expect(revived.constructor).toBe(est.constructor);
                expect(revived.getParams()).toEqual(est.getParams());
                // state parity: re-serializing the revived model must be identical
                expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(est)));
                // behavior parity where an out-of-sample inference path exists
                const replay = runFitted(revived, spec.kind, data);
                if (replay !== null) {
                    expect(replay).toEqual(out1);
                }
            });

            it('setParams after fit resets to a working unfitted estimator', () => {
                const est = spec.create();
                fitAndRun(est, spec.kind, data);
                est.setParams({});
                const out = fitAndRun(est, spec.kind, data);
                expect((out as unknown[]).length).toBe(data.X.length);
            });
        });
    }
}
