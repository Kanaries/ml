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

6. **Optional feature-selection capabilities.** Estimators opt in
   structurally; `BaseEstimator` does not install placeholder properties.
   Linear estimators whose fitted weights map one-to-one to input features
   expose a read-only `coef` getter. Trees, forests, and boosting estimators
   expose normalized, non-negative `featureImportances`. Kernel SVMs and
   estimators whose fitted coefficients refer to expanded/internal features
   do not declare either capability. Use `hasCoefficientCapability()` or
   `hasFeatureImportanceCapability()` before access. The conformance harness
   validates only properties an estimator actually declares. Access before a
   model is fitted may return the estimator's historical empty value or throw
   a fitted-state error; meta-estimators must check the returned length. Models
   serialized before importance metadata was introduced still predict normally
   but throw an explicit refit-required error when importance is requested.

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
- Declared `coef` values must reduce to exactly one finite value per input
  feature. Declared `featureImportances` must have the input feature count,
  contain finite non-negative values, and either sum to one or be all-zero for
  a tree that cannot split.

## Dense, sparse, and text data

`TransformerBase<TInput, TOutput>` is generic so transformers are no longer
restricted to `number[][]`. `Pipeline` accepts raw `string[]` documents at an
explicit text-transformer entry point and can carry either dense matrices or
`CSRMatrix` between steps. Numeric estimators remain free to support only
dense matrices; sparse-aware estimators declare `NumericMatrix` inputs. The
Naive Bayes family supports both dense and CSR input, including inference
without converting a document-term matrix to a dense table.
Because TypeScript generics are erased at runtime, non-dense transformers and
estimators override `acceptedInputKinds` (for example `['text']` or
`['dense', 'csr']`) so Pipeline can reject an incompatible step with a named
boundary error. Components without that declaration default to dense-only.

`CSRMatrix` lives in the public `Data` namespace and provides `shape`, `nnz`,
row access, non-zero row iteration, row selection, dense conversion, and model
codec support. Its constructor validates canonical CSR invariants (monotonic
row pointers, sorted unique in-range columns, and no explicit zeros).
The Wave 0 memory gate runs the browser-compatible public bundle on Node/V8
with a hard 256 MiB process limit. `npm run benchmark:csr:browser` additionally
runs the same 20k × 30k fit/predict workload in headless Chrome/Chromium (set
`CHROME_BIN` when the browser is not in a standard location).

## Serialization format

`estimator.toJSON()` → `{ format: '@kanaries/ml-model', formatVersion: 1, estimator, params, state }`.
Values JSON can't express are wrapped in tag objects (`$num`, `$undef`,
`$map`, `$set`, `$typed`, `$cls`, `$obj`). `loadModel(jsonOrString)` revives
any registered estimator; `Class.fromJSON(json)` additionally enforces the
class.
