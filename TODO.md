# TODO

> **Authoritative roadmap**: [docs/ML_ROADMAP_NEXT.md](docs/ML_ROADMAP_NEXT.md).
> This file is the short-form working checklist. Last synced with the codebase: 2026-07-31
> (114 exported public classes across 20+ modules; Phases 0–2 of the roadmap shipped 2026-07-11/12).

## Done (shipped, previously mistracked here)

Estimator contract (`getParams`/`setParams`/`clone`, `toJSON`/`fromJSON`, conformance suite) ·
Pipeline / ColumnTransformer / FeatureUnion · full metrics & preprocessing fill-out ·
model selection (KFold family, Grid/RandomizedSearchCV, crossValidate, learning/validation curves) ·
RandomForest · Bagging (clf) · AdaBoost · GradientBoosting · XGBoost · Voting · Stacking ·
SVC/SVR/NuSVC/NuSVR/LinearSVC/LinearSVR/OneClassSVM · full Naive Bayes family ·
KMeans/MiniBatchKMeans/DBSCAN/HDBSCAN/OPTICS/MeanShift/Agglomerative/Spectral ·
GaussianMixture (+Bayesian) · LDA/QDA · MLP C/R · BernoulliRBM · t-SNE/LLE/MDS/SpectralEmbedding ·
PCA/SparsePCA/TruncatedSVD · SGD family · Perceptron · LabelPropagation/Spreading ·
OvR/OvO · MultiOutput C/R · CalibratedClassifierCV · IsotonicRegression · Dummy baselines ·
`datasets.make*` generators · Web Worker `asyncMode`.

## Active — Phase 3 backlog (roughly priority-ordered)

### Anomaly detection (completes an advertised feature area)
- [ ] LocalOutlierFactor (reuse KDTree/BallTree)
- [ ] EllipticEnvelope (needs MinCovDet, see covariance below)

### Symmetry gaps (small, high value)
- [ ] BaggingRegressor (classifier exists)
- [ ] ExtraTreesClassifier / ExtraTreesRegressor (forest of existing ExtraTree)
- [ ] Isomap (manifold family four-of-five done; reuse neighbors + MDS infra)
- [ ] crossValPredict

### Feature selection
- [ ] chi2, fClassif, mutual information scores
- [ ] SelectFromModel
- [ ] RFE / RFECV
- [ ] SequentialFeatureSelector

### Robust & probabilistic linear models
- [ ] HuberRegressor, RANSACRegressor, TheilSenRegressor
- [ ] BayesianRidge, ARDRegression
- [ ] PoissonRegressor / GammaRegressor / TweedieRegressor (GLMs)
- [ ] QuantileRegressor

### Decomposition
- [ ] KernelPCA
- [ ] FastICA
- [ ] NMF
- [ ] IncrementalPCA
- [ ] FactorAnalysis
- [ ] LatentDirichletAllocation

### Text feature extraction (unlocks browser NLP with existing MultinomialNB/ComplementNB)
- [ ] CountVectorizer
- [ ] TfidfVectorizer

### Remaining estimator families
- [ ] KernelRidge · KernelDensity
- [ ] GaussianProcessRegressor / GaussianProcessClassifier (+ kernels)
- [ ] covariance module: EmpiricalCovariance, LedoitWolf, MinCovDet, GraphicalLasso
- [ ] cross_decomposition: PLSRegression, CCA
- [ ] random_projection: Gaussian / Sparse
- [ ] Birch, AffinityPropagation, BisectingKMeans
- [ ] SelfTrainingClassifier · ClassifierChain / RegressorChain
- [ ] IterativeImputer · SplineTransformer · TargetEncoder
- [ ] inspection: permutationImportance, partialDependence
- [ ] HistGradientBoosting C/R (pairs with Phase 4 typed-array work)

## Phase 4 — JS differentiation (after Phase 3 core)

- [ ] Typed-array internals for hot paths + benchmark suite
- [ ] `partialFit` (SGD, MiniBatchKMeans, NB family)
- [ ] sklearn-model import (Python export helper + JS loader)
- [ ] Bundle-size budget + subpath exports
- [ ] Optional WASM kernels

## Explicitly out of scope (decided, do not re-add)

- CNN / RNN / LSTM — deep learning is TensorFlow.js territory per the README positioning; MLP is the boundary.
- ARIMA / exponential smoothing / seasonal decomposition — sklearn deliberately excludes time series (statsmodels territory); would dilute the sklearn-parity promise. Revisit only as a separate package.
- Apriori / association rules — mlxtend territory, not sklearn.
- Genetic algorithms / particle swarm optimization, image preprocessing — outside the sklearn surface.
