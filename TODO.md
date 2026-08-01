# TODO — Phase 3 Check Items

> Working checklist, one checkbox per check item. Progress is reported as check-item counts
> (e.g. "Wave A: 9/16") — no time estimates. Last synced with the codebase: 2026-08-01.
> Historical roadmap: [docs/ML_ROADMAP_NEXT.md](docs/ML_ROADMAP_NEXT.md) (its §1–2 are a
> pre-Phase-0 snapshot; Phases 0–2 shipped 2026-07-11/12 — estimator contract & serialization,
> Pipeline/ColumnTransformer/FeatureUnion, metrics/preprocessing/model-selection fill-out,
> GMM, MLP, SVR family, LDA/QDA, SGD family, Voting/Stacking, calibration, and more).

Acceptance per item — estimator: implementation + sklearn parity fixture + conformance
suite + docs entry. Utility: implementation + numeric parity test + docs. Architecture:
stated DoD + all existing tests green.

## Wave 0 — Architecture prerequisites (4 items)

- [x] 0-1 `featureImportances`/`coef` as an **optional capability** protocol — implemented only by estimators with the semantics (linear → `coef`, trees/forests/boosting → `featureImportances`; kernel SVMs etc. explicitly opt out); conformance checks only declared capabilities; SelectFromModel/RFE gate on capability detection
- [x] 0-2 Forest skeleton generalization: RandomForest takes a parameterized base estimator, behavior unchanged (prereq for ExtraTrees)
- [x] 0-3 Data-abstraction upgrade: string-feature input path + minimal CSR sparse representation; Pipeline/TransformerBase unpinned from `number[][]`; NB family accepts sparse (prereq for text route; memory benchmark: 20k docs × 30k vocab in-browser)
- [x] 0-4 `scripts/coverage-vs-sklearn` script (sklearn 1.5 `all_estimators()` denominator, alias-deduped; coverage numbers come only from this)

## Wave A — Promise gaps + symmetry (16 items)

Estimators (10):
- [x] LocalOutlierFactor
- [x] EllipticEnvelope
- [x] MinCovDet (FAST-MCD, high complexity)
- [x] Isomap (neighbor graph + shortest paths + spectral embedding; not assembled from MDS)
- [x] BaggingRegressor
- [x] ExtraTreesClassifier (needs 0-2)
- [x] ExtraTreesRegressor (needs 0-2)
- [x] SelectFromModel (meta-transformer — estimator acceptance; needs 0-1)
- [x] RFE (meta-transformer — estimator acceptance; needs 0-1)
- [x] RFECV (meta-transformer — estimator acceptance; needs 0-1)

Utilities (6):
- [x] crossValPredict
- [x] chi2
- [x] fClassif
- [x] mutualInfoClassif (k-NN entropy estimator)
- [x] mutualInfoRegression
- [x] sklearn naming aliases (DBSCAN, HDBSCAN, KNeighborsClassifier, …; old names kept)

## Wave B — Text route + high-demand decomposition (13 items)

- [x] CountVectorizer (needs 0-3)
- [x] TfidfTransformer (needs 0-3)
- [x] TfidfVectorizer (needs 0-3)
- [x] KernelPCA
- [x] FastICA
- [x] NMF
- [x] IncrementalPCA
- [x] SelfTrainingClassifier
- [x] ClassifierChain
- [x] RegressorChain
- [x] IterativeImputer (iterative refitting, medium cost)
- [x] GroupShuffleSplit
- [x] StratifiedGroupKFold

## Wave C — Statistical & kernel long tail (35 items)

Robust/probabilistic linear (6):
- [ ] HuberRegressor
- [ ] RANSACRegressor
- [ ] TheilSenRegressor (subsample combinatorics, medium cost)
- [ ] QuantileRegressor (needs LP solver, highest cost in this wave)
- [ ] BayesianRidge
- [ ] ARDRegression

GLMs (3) — per-item solver/link/regularization alignment, no shared-IRLS shortcut:
- [ ] PoissonRegressor
- [ ] GammaRegressor
- [ ] TweedieRegressor

Kernel methods (2):
- [ ] KernelRidge (Cholesky solve on kernel matrix; reuses kernel fns, not SMO)
- [ ] KernelDensity

Clustering (3):
- [ ] Birch
- [ ] AffinityPropagation (O(n²) memory — document sample-size ceiling)
- [ ] BisectingKMeans

Decomposition (2):
- [ ] FactorAnalysis
- [ ] LatentDirichletAllocation

Covariance (5):
- [ ] EmpiricalCovariance
- [ ] ShrunkCovariance
- [ ] LedoitWolf
- [ ] OAS
- [ ] GraphicalLasso

Cross-decomposition (2):
- [ ] PLSRegression
- [ ] CCA

Random projection (2):
- [ ] GaussianRandomProjection
- [ ] SparseRandomProjection

Text extras (3):
- [ ] HashingVectorizer
- [ ] DictVectorizer
- [ ] FeatureHasher

Other (7):
- [ ] TransformedTargetRegressor
- [ ] NearestNeighbors (unsupervised query wrapper over KDTree/BallTree)
- [ ] permutationImportance
- [ ] partialDependence
- [ ] SplineTransformer
- [ ] TargetEncoder
- [ ] MultiLabelBinarizer

## Wave D — Heavy projects (4 items; excluded from the Phase 3 exit gate)

- [ ] GaussianProcessRegressor (kernel system accepted as part of this item)
- [ ] GaussianProcessClassifier
- [ ] HistGradientBoostingClassifier (pair with Phase 4 typed-array work)
- [ ] HistGradientBoostingRegressor (pair with Phase 4 typed-array work)

## Phase 3 exit gate

`src/__test__/e2e/` scenario corpus: 10 frozen named scenarios, each a spec of an explicit
algorithm subset + dataset + workflow (frozen at Wave A start; specs exclude algorithms that
are Deferred — e.g. faces decomposition runs PCA/NMF/FastICA only, manifold comparison runs
standard LLE only). Parity is 1:1 against the frozen spec, not against full official example
pages. Exit = 10/10 e2e green + Wave 0/A/B at 100% + Wave C ≥ 80% (trimmed Wave C items move
to Deferred with a decision note; specs referencing them get revised in the same commit).

## Phase 4 — JS differentiation (after Phase 3)

- [ ] Typed-array internals for hot paths + benchmark suite
- [ ] `partialFit` (SGD, MiniBatchKMeans, NB family)
- [ ] sklearn-model import (Python export helper + JS loader)
- [ ] Bundle-size budget + subpath exports
- [ ] Optional WASM kernels

## Deferred (enumerated, not scheduled; promoting one requires a decision note)

PLSCanonical · PLSSVD · OutputCodeClassifier · NCA · KNeighborsTransformer ·
RandomTreesEmbedding · FeatureAgglomeration · DictionaryLearning · Lars · LassoLars ·
OrthogonalMatchingPursuit · PassiveAggressive C/R · HalvingGridSearchCV ·
HalvingRandomSearchCV · LeavePOut · LeaveOneGroupOut · permutationTestScore ·
GraphicalLassoCV

## Explicitly out of scope (decided, do not re-add)

- CNN / RNN / LSTM — deep learning is TensorFlow.js territory per the README positioning; MLP is the boundary.
- ARIMA / exponential smoothing / seasonal decomposition — sklearn deliberately excludes time series; would dilute the sklearn-parity promise. Revisit only as a separate package.
- Apriori / association rules — mlxtend territory, not sklearn.
- Genetic algorithms / particle swarm optimization, image preprocessing — outside the sklearn surface.
- `*CV` estimator variants (RidgeCV, LassoCV, …) — not strictly equivalent to GridSearchCV composition (efficient LOOCV/GCV paths, `alphaPerTarget` semantics), but at this library's target data scale the composition is an acceptable approximation; docs show the equivalent. Reopen on real user demand.
- UMAP — not sklearn (umap-learn), high cost. Watch-listed.
