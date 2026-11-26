# sklearn parity checklist

This list highlights scikit-learn algorithms and APIs that are not yet implemented in `@kanaries/ml`, based on the current README-supported set. Use it to track feature parity work.

Current coverage (from README): tree models, KNN utilities, linear/logistic regression, SVC family, Bernoulli/Categorical NB, KMeans/DBSCAN/OPTICS/MeanShift/HDBSCAN clustering, PCA, SpectralEmbedding/MDS/LLE/TSNE manifold learners, IsolationForest, and AdaBoost. 【F:README.md†L51-L64】

## Linear models and generalized linear models
- [ ] Ridge and Lasso regressors (`Ridge`, `Lasso`, `ElasticNet`, `RidgeClassifier`)
- [ ] Stochastic gradient solvers (`SGDClassifier`, `SGDRegressor`, `Perceptron`, `PassiveAggressiveClassifier`)
- [ ] Linear discriminant analysis and quadratic discriminant analysis (`LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`)

## Support Vector Machines
- [ ] Regression and anomaly detection variants (`SVR`, `NuSVR`, `OneClassSVM`)

## Naive Bayes
- [ ] Gaussian and count-based variants (`GaussianNB`, `MultinomialNB`, `ComplementNB`)

## Ensemble methods
- [ ] Bagging-style ensembles (`RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `BaggingClassifier`)
- [ ] Boosting/gradient ensembles (`GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`)
- [ ] Voting and stacking meta-estimators (`VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`)

## Neighbors
- [ ] Radius-based neighbors and regressors (`RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`, `KNeighborsRegressor`, `NearestCentroid`)

## Clustering and mixture models
- [ ] Hierarchical and graph-based clustering (`AgglomerativeClustering`, `FeatureAgglomeration`, `SpectralClustering`, `SpectralBiclustering`, `SpectralCoclustering`)
- [ ] Other clustering algorithms (`Birch`, `AffinityPropagation`, `GaussianMixture`, `BayesianGaussianMixture`, `MiniBatchKMeans`)

## Decomposition and topic models
- [ ] Independent/non-negative factorization (`FastICA`, `NMF`) and latent factor models (`FactorAnalysis`, `LatentDirichletAllocation`)
- [ ] Incremental/online variants (`IncrementalPCA`, `MiniBatchSparsePCA`)

## Manifold learning and embeddings
- [ ] Additional embeddings (`Isomap`) and graph-based embeddings (`LaplacianEigenmap`/`SpectralEmbedding` already exists but lacks adjacency helpers)

## Neural networks
- [ ] Multi-layer perceptrons (`MLPClassifier`, `MLPRegressor`) and feature transformers (`MLPClassifier`-style embedding)

## Preprocessing and feature extraction
- [ ] Data scaling and normalization (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `Normalizer`)
- [ ] Encoding utilities (`OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `Binarizer`)
- [ ] Imputation and missing-value handling (`SimpleImputer`, `KNNImputer`) and feature selectors (`VarianceThreshold`, `SelectKBest`)

## Model selection, validation, and composition
- [ ] Dataset splitting and cross-validation helpers (`train_test_split`, `StratifiedKFold`, `cross_val_score`)
- [ ] Hyperparameter search utilities (`GridSearchCV`, `RandomizedSearchCV`)
- [ ] Pipeline and column composition (`Pipeline`, `FeatureUnion`, `ColumnTransformer`)

## Metrics
- [ ] Classification, regression, and clustering metrics (`accuracy_score`, `precision_recall_fscore_support`, `roc_auc_score`, `r2_score`, `adjusted_rand_score`)
- [ ] Model evaluation curves (`roc_curve`, `precision_recall_curve`, `confusion_matrix`)
