# Phase 2 Independent Review — Findings & Resolutions

Reviewer: Claude Opus 4.8 subagent (independent; the codex CLI review of this
diff hung on the 13k-line change and was terminated). Scope: commit `f895e65`
on feat/phase2-estimators — 21 new estimators across 9 modules. Findings were
empirically reproduced by the reviewer before reporting. Date: 2026-07-12.

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| 1 | P1 | `SGDClassifier`/`Perceptron` defined `predictProba` as an always-present method that throws for margin losses (`hinge`/`perceptron`/`squaredHinge`), while `CalibratedClassifierCV`, `OneVsRestClassifier`'s `binaryScores`, and soft-capability probes detect support via `typeof est.predictProba === 'function'`. Calibrating a **default** SGDClassifier — the textbook Platt-scaling use case — crashed; OvR with a hinge base crashed on predict. | **Fixed** — for non-probabilistic losses the capability is now genuinely absent (non-enumerable own `predictProba: undefined`, sklearn `available_if` semantics; invisible to serialization/setParams and re-derived through the constructor on revive). `binaryScores` additionally prefers `decisionFunction` first, mirroring sklearn's `_predict_binary` and matching `OneVsOneClassifier`. |
| 2 | P2 | Same root cause in `StackingClassifier` with the default `stackMethod: 'auto'`: `resolveStackMethod` picked `predictProba` whenever the method existed, so any hinge/perceptron SGD base made `fit` throw instead of falling back to `decisionFunction`. | **Fixed** by the same capability change — `auto` now resolves to `decisionFunction` for margin-loss members (no stacking-code change needed). |

Areas explicitly verified as correct by the reviewer (formulas checked against
sklearn/libsvm/scipy, several numerically reproduced): GMM/BGMM EM and
variational updates incl. aic/bic parameter counts; MLP backprop/Adam/
Nesterov/schedules/early-stopping; SVR/NuSVR/OneClassSVM SMO duals; NN-chain
Lance-Williams and threshold cutting; spectral Laplacian/Jacobi/NJW;
MiniBatchKMeans updates and stopping; LDA svd/eigen chains; QDA; SGD optimal
schedule, cumulative L1, subgradients; stacking OOF leak-freedom; voting
argmax/ties; Platt targets and Newton; PAVA; raw-props getParams everywhere;
fit-local RNGs; serialization round-trips.

Regression coverage: `src/__test__/phase2.review-fixes.test.ts`.
All fixes verified by the full suite (`yarn jest`) and typecheck.
