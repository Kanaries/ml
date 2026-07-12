# Phase 0 CodeX Review — Findings & Resolutions

Reviewer: codex-cli 0.144.1 (independent), scope: commits `2806a86` + `9dcb2f9`
(estimator contract, serialization codec, ~60-estimator retrofit, conformance harness).
Date: 2026-07-11.

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| 1 | P1 | Prototype injection via crafted model JSON: `decodeValue`/`loadModel` assign attacker-controlled keys (`__proto__`, `constructor`) onto objects and revived instances. | **Fixed** — `assertSafeKey()` rejects `__proto__`/`constructor`/`prototype` at every decode/assignment site; regression tests added. |
| 2 | P1 | `BaseEstimator.setParams` rebuild used `Object.keys` + `Object.assign`, so non-enumerable hidden fields (tree RNGs) were neither deleted nor copied — stale RNG survives a param change. | **Fixed** — rebuild now uses `getOwnPropertyNames` + property descriptors, preserving hidden-field semantics. |
| 3 | P1 | Nested estimators serialized via generic `$cls` (state-only, `Object.create` revival) lose constructor-installed non-enumerable fields; a revived nested tree turns its RNG into an enumerable function on refit → next `toJSON` throws. | **Fixed** — estimator instances now encode as full `$est` envelopes and revive through `loadModel` (constructor runs, hidden fields restored). Tree/ensemble `fit()` RNG reseeds switched to `defineHiddenField` as belt-and-braces. Harness gained a revive→refit parity check. |
| 4 | P1 | `HDBScan.getParams()` returned the *resolved* `min_samples` (defaults to `min_cluster_size`), so `setParams({min_cluster_size})` silently pinned the old dependent value; grid search evaluates wrong models. Same class of bug found in `OPTICS.max_eps` (defaults to `eps`). | **Fixed** — both classes store and return the raw prop and resolve lazily at fit time; regression tests assert setParams-vs-fresh-construction equivalence. |
| 5 | P1 | Claimed: refitting a revived seeded tree diverges from refitting the original (advanced vs fresh RNG). | **Not a bug** — tree `fit()` reseeds from `randomState` on every call, so refits are reproducible on both sides; now also covered by the harness revive→refit check. The underlying hidden-field loss was real and is fixed as #3. |
| 6 | P2 | Typed-array elements bypassed the codec: `NaN`/`±Infinity` → JSON `null` → silently revived as `0`. | **Fixed** — typed-array elements are encoded per-element through the codec; round-trip test added. |
| 7 | P2 | `-0` not preserved through JSON text. | **Fixed** — `-0` gets a `$num` tag; asserted in tests. |
| 8 | P2 | `setParams` key validation used `key in known`, accepting inherited names (`constructor`, `toString`). | **Fixed** — own-property check; test added. |
| 9 | P2 | `clone()` did not deep-clone estimators inside `Map`/`Set`/null-prototype containers. | **Fixed** — recursion extended to `Map`, `Set`, and null-prototype objects; test added. |

All fixes verified by the full suite (`yarn jest`) and typecheck.
