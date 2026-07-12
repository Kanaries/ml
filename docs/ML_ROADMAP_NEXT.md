# ML Package Roadmap — Gap Analysis vs scikit-learn & Next Phases

_Last updated: July 11, 2026 (supersedes the March 2026 roadmap; all of its items shipped)._

## 1. Current State

~60 estimators across 14 families, with sklearn-parity fixture tests as the established quality bar:

| Family | Implemented |
|---|---|
| Linear | LinearRegression, LogisticRegression, Polynomial, Ridge, RidgeClassifier, Lasso, ElasticNet |
| Trees | DecisionTree C/R, ExtraTree C/R |
| Ensemble | RandomForest C/R, Bagging, AdaBoost C/R, GradientBoosting C/R, IsolationForest, XGBoost C/R |
| SVM | SVC (real SMO), NuSVC, LinearSVC, LinearSVR |
| Naive Bayes | Gaussian, Multinomial, Bernoulli, Complement, Categorical (complete family) |
| Clustering | KMeans(+kmeans++), DBSCAN, HDBSCAN, OPTICS, MeanShift |
| Neighbors | KNN, KNeighborsRegressor, RadiusNeighbors C/R, NearestCentroid, KDTree, BallTree |
| Decomposition | PCA, SparsePCA, TruncatedSVD |
| Manifold | t-SNE, LLE, MDS, SpectralEmbedding |
| Semi-supervised | LabelPropagation, LabelSpreading |
| Neural | BernoulliRBM only |
| Preprocessing | Standard/MinMax/MaxAbs scalers, Normalizer, Label/Ordinal/OneHot encoders, Binarizer, SimpleImputer, VarianceThreshold, SelectKBest, fRegression |
| Model selection | KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, crossValScore |
| Metrics | accuracy, P/R/F1 (+support), MSE, R², confusion matrix, ROC/AUC, PR curve, adjustedRandScore, distance metrics |

## 2. Gap Analysis vs scikit-learn

The gaps fall into four layers. Ordered by leverage, not by count:

### 2.1 Architecture layer (highest leverage — blocks everything else)

- **No Pipeline / ColumnTransformer / FeatureUnion.** Users cannot compose preprocessing + estimator into one object; GridSearchCV cannot tune a whole workflow. This is the single biggest usability gap vs sklearn.
- **No model serialization.** No `toJSON`/`fromJSON` anywhere in `src/`. In a JS library this is worse than in Python (no pickle equivalent): train-once/deploy-in-browser is the library's core value proposition and currently impossible.
- **Inconsistent estimator contract.** `ClassifierBase` only mandates `fit`/`predict`. `predictProba` exists on ~5 estimators, `sampleWeight` on ~4. No `getParams`/`setParams`/`clone`, so meta-estimators (search, pipelines, ensembles-of-anything) can't treat estimators generically.
- **`number[][]` everywhere.** No typed-array path, no sparse-matrix support. Caps practical dataset size well below what JS can actually handle.

### 2.2 Missing estimators (by family, roughly priority-ordered within each)

- **Mixture models**: GaussianMixture, BayesianGaussianMixture — most-requested absent family.
- **Neural**: MLPClassifier / MLPRegressor (RBM alone is not a usable NN story).
- **SVM**: SVR / NuSVR (kernel regression — SMO core already exists to build on), OneClassSVM.
- **Clustering**: AgglomerativeClustering (+ dendrogram/linkage), SpectralClustering, MiniBatchKMeans, Birch, BisectingKMeans, AffinityPropagation.
- **Discriminant analysis**: LDA, QDA.
- **Linear (online/robust)**: SGDClassifier/SGDRegressor, Perceptron, PassiveAggressive, BayesianRidge, HuberRegressor, RANSAC, TheilSen, QuantileRegressor, GLMs (Poisson/Gamma/Tweedie).
- **Meta-estimators**: VotingClassifier/Regressor, StackingClassifier/Regressor, OneVsRest/OneVsOne, MultiOutputClassifier/Regressor, CalibratedClassifierCV, SelfTrainingClassifier, Dummy baselines.
- **Ensemble**: HistGradientBoosting C/R (binned, LightGBM-style — matters for perf story even with XGBoost present).
- **Decomposition**: NMF, FastICA, KernelPCA, FactorAnalysis, IncrementalPCA, LatentDirichletAllocation.
- **Manifold**: Isomap (UMAP as a stretch goal — not sklearn, but expected by users).
- **Outlier detection**: LocalOutlierFactor, EllipticEnvelope.
- **Other**: GaussianProcessRegressor/Classifier, KernelRidge, KernelDensity, IsotonicRegression.

### 2.3 Utility layer

- **Metrics**: log loss, MAE/MAPE/median AE, balanced accuracy, Matthews corrcoef, Cohen's kappa, classification report, multiclass ROC-AUC, brier; clustering: silhouette, Davies–Bouldin, Calinski–Harabasz, NMI/AMI, homogeneity/completeness/v-measure; pairwise kernel/distance matrix API.
- **Preprocessing**: RobustScaler, PowerTransformer, QuantileTransformer, KBinsDiscretizer, PolynomialFeatures, SplineTransformer, FunctionTransformer, TargetEncoder, KNNImputer, IterativeImputer, MissingIndicator, LabelBinarizer/MultiLabelBinarizer.
- **Feature selection**: chi², f_classif, mutual information, RFE/RFECV, SelectFromModel, SequentialFeatureSelector.
- **Model selection**: train_test_split parity options, ShuffleSplit, GroupKFold, TimeSeriesSplit, RepeatedKFold, LeaveOneOut, cross_validate (multi-metric), learning_curve, validation_curve, HalvingGridSearch.
- **Feature extraction**: CountVectorizer, TfidfVectorizer, DictVectorizer, FeatureHasher.
- **Datasets**: make_classification / make_regression / make_blobs / make_moons / make_circles + a couple of bundled toy datasets (also great for docs/SEO examples).
- **Inspection**: permutation_importance, partial dependence.

### 2.4 JS-specific opportunities (sklearn can't have these)

- Web Worker / async training (`asyncMode` exists — generalize it), streaming `partialFit`.
- Typed arrays → optional WASM/SIMD kernels for hot loops (distance matrices, SMO, tree splits); WebGPU as far-future.
- Tree-shakeable subpath exports so a browser bundle with one estimator stays small.
- Import path for sklearn-trained models (accept a documented JSON schema exported from Python) — pairs with serialization.

## 3. Phased Plan

### Phase 0 — Foundation reset (✅ completed 2026-07-11, branch feat/estimator-contract)

1. Land/merge the algorithm-remediation branch so new work builds on correct implementations.
2. Define the estimator contract: `getParams`/`setParams`/`clone`, `predictProba`/`decisionFunction` where meaningful, `sampleWeight` in `fit`, and `toJSON`/`fromJSON` (versioned schema). Retrofit all ~60 estimators. Breaking API changes are acceptable (internal-usage only).
3. Add a `checkEstimator`-style conformance test that every estimator must pass (fit-idempotence, clone-equivalence, serialize→deserialize→predict parity).

**Exit criteria**: every existing estimator passes conformance; any model can round-trip through JSON.

### Phase 1 — Workflow completeness (✅ completed 2026-07-11, same branch)

1. `Pipeline`, `ColumnTransformer`, `FeatureUnion` (built on the Phase 0 contract).
2. Metrics fill-out (§2.3 list), `classificationReport`.
3. Preprocessing fill-out: RobustScaler, PowerTransformer, QuantileTransformer, PolynomialFeatures, KBinsDiscretizer, KNNImputer, LabelBinarizer.
4. Model-selection fill-out: ShuffleSplit, GroupKFold, TimeSeriesSplit, cross_validate, learning/validation curves.
5. `datasets.make*` generators (unblocks docs, tests, and SEO example pages).

**Exit criteria**: the canonical sklearn tutorial (load → split → pipeline(scale+model) → grid search → report) is reproducible 1:1 in JS.
**Status**: exit criteria met — see `src/__test__/e2e.tutorial.test.ts`. Independent CodeX reviews ran per phase (`docs/reviews/`); all P0–P2 findings fixed. 1235 tests green.

### Phase 2 — High-demand estimators (✅ completed 2026-07-12, branch feat/phase2-estimators)

GaussianMixture (+Bayesian), MLPClassifier/Regressor, SVR/NuSVR/OneClassSVM (reuse SMO core), AgglomerativeClustering + SpectralClustering + MiniBatchKMeans, LDA/QDA, SGD family + Perceptron, Voting/Stacking, OneVsRest/OneVsOne/MultiOutput wrappers, CalibratedClassifierCV, Dummy baselines. All with sklearn-parity fixtures.

### Phase 3 — Depth and breadth

NMF, FastICA, KernelPCA, FactorAnalysis, IncrementalPCA; Isomap; LocalOutlierFactor, EllipticEnvelope; Gaussian Processes; KernelRidge, KernelDensity, IsotonicRegression; robust/GLM linear family; HistGradientBoosting; text feature extraction (Count/TfidfVectorizer); RFE/SelectFromModel/mutual information; permutation importance + partial dependence.

### Phase 4 — JS differentiation & performance

Typed-array internals for hot paths → benchmark suite vs current `number[][]`; Web Worker training API; `partialFit` for SGD/MiniBatchKMeans/NB family; optional WASM kernels; sklearn-model import (Python export helper + JS loader); bundle-size budget + subpath exports; docs/benchmark pages feeding the SEO/AEO initiative.

## 4. Sequencing Rationale

- Serialization and the estimator contract come first because every estimator added before them increases retrofit cost, and because "train anywhere, run in the browser" is the differentiator vs sklearn — not algorithm count.
- Pipeline before new estimators: 60 estimators that can't compose lose to 20 that can.
- Parity-fixture testing (already established in CI) remains the acceptance bar for every new estimator; vacuous-threshold tests are not accepted.

## 5. Coverage Target

sklearn's commonly-used surface is roughly 130 estimators + 150 utilities. Today we cover ≈35–40% of the estimator surface and ≈25% of utilities. End of Phase 2 target: ≈70% of the estimators that appear in the top-100 sklearn doc pages by traffic; end of Phase 3: "relatively complete" — a user following any mainstream sklearn tutorial finds an equivalent here.
