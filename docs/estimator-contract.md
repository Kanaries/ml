# Estimator Contract (Phase 0)

Every estimator in `@kanaries/ml` implements a common contract defined in
[`src/base/estimator.ts`](../src/base/estimator.ts). This enables generic
meta-estimators (pipelines, search), cloning, and JSON model persistence.

## Requirements for every estimator class

1. **Base class.** Extend the matching base:
   - `ClassifierBase` — `fit(X, y, sampleWeight?)` + `predict(X)`; optional `predictProba` / `decisionFunction`.
   - `RegressorBase` — `fit(X, y, sampleWeight?)` + `predict(X)`; `score` = R².
   - `ClusterBase` — `fitPredict(X, sampleWeights?)`.
   - `TransformerBase` — `fit(X, y?)` + `transform(X)`; `fitTransform` provided; optional `inverseTransform`.
   - `OutlierBase` — `fit(X)` + `predict(X)`; `fitPredict` provided.
   - Estimators that fit none of these signatures (embeddings with only
     `fitTransform`, label encoders operating on 1-D arrays, search wrappers)
     extend `BaseEstimator` directly and keep their established public API.

2. **Props-object constructor.** The canonical constructor takes a single
   optional props object. Classes that historically took positional arguments
   keep the positional form as a **deprecated overload** (see `KMeans` for the
   pattern) — never break existing call sites.

3. **`getParams()`.** Return exactly the props-object shape (same keys, current
   values). Do not include fitted state. If a param was transformed at
   construction (e.g. copied arrays), return the semantically equivalent value.

4. **Registration.** Immediately after the class definition:
   `registerEstimator('ClassName', ClassName);` using the exported class name.
   Helper classes stored in fitted state (tree nodes, KD-trees, nested
   estimators from other modules are already covered by their own
   registration) must be registered with
   `registerSerializableClass('module.ClassName', ClassName);`.

5. **Serializable state.** All instance fields must be plain data, `Map`,
   `Set`, typed arrays, or registered class instances. Function-valued fields
   are not serializable: params accepting callbacks must also accept a
   built-in's **name** (string) and store the name, resolving the function
   lazily.

Provided generically by `BaseEstimator` (do not override without reason):
`setParams` (validates keys, rebuilds through the constructor, resets fitted
state), `clone`, `toJSON`, `Class.fromJSON`, and the top-level `loadModel`.

## Conformance tests

Each module has `src/<module>/__test__/conformance.test.ts` calling
`runEstimatorConformance` from
[`src/__test__/conformance/harness.ts`](../src/__test__/conformance/harness.ts)
with one spec per estimator:

```ts
runEstimatorConformance([
    { name: 'KMeans', kind: 'cluster', create: () => new KMeans({ n_clusters: 3, random_state: 42 }) },
]);
```

- `create()` must return a **seeded** instance whenever the estimator has any
  randomness (pass `random_state`/`randomState`); use `nonDeterministic: true`
  only when the class offers no seed control.
- Pick the `dataset` the estimator can actually learn (`binary` for
  binary-only classifiers, `counts` for Multinomial/Complement NB,
  `binaryFeatures` for Bernoulli-style models, ...).
- Estimators whose shape the harness doesn't cover get a small hand-written
  test covering the same points: registration, params round-trip, clone,
  serialize → revive → equal state/behavior.

## Serialization format

`estimator.toJSON()` → `{ format: '@kanaries/ml-model', formatVersion: 1, estimator, params, state }`.
Values JSON can't express are wrapped in tag objects (`$num`, `$undef`,
`$map`, `$set`, `$typed`, `$cls`, `$obj`). `loadModel(jsonOrString)` revives
any registered estimator; `Class.fromJSON(json)` additionally enforces the
class.
