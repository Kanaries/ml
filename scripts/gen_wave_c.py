"""Generate deterministic sklearn reference data for Phase 3 Wave C."""
import json
from pathlib import Path

import numpy as np
import sklearn
from scipy.integrate import quad
from scipy.special import gamma as gamma_function
from sklearn.cluster import AffinityPropagation, Birch, BisectingKMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.covariance import EmpiricalCovariance, GraphicalLasso, LedoitWolf, OAS, ShrunkCovariance
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import FactorAnalysis, LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, GammaRegressor, HuberRegressor, LinearRegression,
    PoissonRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor, TweedieRegressor,
)
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, SplineTransformer, TargetEncoder
from sklearn.random_projection import johnson_lindenstrauss_min_dim


if sklearn.__version__ != '1.9.0':
    raise RuntimeError(f'Wave C fixtures require scikit-learn 1.9.0, found {sklearn.__version__}')


kernel_x = np.array([[-2, 1], [-1, 0], [0, 1], [1, 0], [2, 1]], dtype=float)
kernel_y = np.array([-1.5, -.2, 1, 2.2, 4.1])
kernel_query = np.array([[-1.5, .5], [.5, .25], [2.5, 1]])
kernel_ridge = KernelRidge(alpha=.5, kernel='rbf', gamma=.7).fit(kernel_x, kernel_y)

density_x = np.array([[-1, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
density_query = np.array([[0, 0], [.5, .5], [2, 0]], dtype=float)
kde_gaussian = KernelDensity(bandwidth=.6, kernel='gaussian').fit(density_x)
kde_epanechnikov = KernelDensity(bandwidth=1.5, kernel='epanechnikov').fit(density_x)
cosine_dimensions = [1, 2, 5, 23, 50]
cosine_origin_scores = []
for dimension in cosine_dimensions:
    unit_ball = np.pi ** (dimension / 2) / gamma_function(dimension / 2 + 1)
    radial = quad(lambda radius: radius ** (dimension - 1) * np.cos(np.pi * radius / 2), 0, 1, epsabs=1e-14)[0]
    cosine_origin_scores.append(float(-np.log(dimension * unit_ball * radial)))

covariance_x = np.array([
    [0, 1, 2], [1, 0, 1], [2, 1, 0], [3, 2, 1], [4, 1, 3], [5, 3, 2], [6, 2, 4], [7, 4, 3],
], dtype=float)
empirical = EmpiricalCovariance().fit(covariance_x)
shrunk = ShrunkCovariance(shrinkage=.2).fit(covariance_x)
ledoit = LedoitWolf().fit(covariance_x)
oas = OAS().fit(covariance_x)
graphical = GraphicalLasso(alpha=.1, max_iter=500, tol=1e-6).fit(covariance_x)

documents = ['The quick brown fox', 'the fox jumped high', 'brown dog and fox']
hashing = HashingVectorizer(n_features=16, norm=None, alternate_sign=True)
hashing_matrix = hashing.transform(documents).toarray()
hashing_default = HashingVectorizer(n_features=16).transform(documents).toarray()

dict_rows = [{'city': 'London', 'temp': 12}, {'city': 'Dubai', 'temp': 33}, {'city': 'London', 'temp': 18}]
dict_vectorizer = DictVectorizer(sparse=False).fit(dict_rows)

hasher_rows = [{'city=London': 1, 'temp': 12}, {'city=Dubai': 1, 'temp': 33}]
feature_hasher = FeatureHasher(n_features=16, input_type='dict', alternate_sign=True)

cluster_x = np.array([[-5, -5], [-5.2, -4.8], [-4.8, -5.1], [0, 0], [.2, -.1], [-.2, .1], [5, 5], [5.1, 4.8], [4.8, 5.2]], dtype=float)
birch = Birch(threshold=.5, n_clusters=None).fit(cluster_x)
affinity = AffinityPropagation(damping=.7, preference=-20, random_state=0, max_iter=500, convergence_iter=20).fit(cluster_x)
bisecting = BisectingKMeans(n_clusters=3, random_state=0, n_init=10, max_iter=300, tol=1e-4, bisecting_strategy='biggest_inertia').fit(cluster_x)
birch_tree_x = np.array([[group * 1.5 + delta, (group % 2) * 2 + delta * .5] for group in range(6) for delta in [-.08, 0, .08]], dtype=float)
birch_tree = Birch(threshold=.12, branching_factor=2, n_clusters=3).fit(birch_tree_x)

factor_x = np.array([[np.sin(i / 7) + .1 * np.cos(i), np.cos(i / 9) + .05 * np.sin(2 * i), .7 * np.sin(i / 7) - .4 * np.cos(i / 9), -.2 * np.sin(i / 7) + .9 * np.cos(i / 9)] for i in range(100)], dtype=float)
factor = FactorAnalysis(n_components=2, svd_method='lapack', tol=1e-4, max_iter=1000).fit(factor_x)
factor_varimax = FactorAnalysis(n_components=2, svd_method='lapack', rotation='varimax', tol=1e-4, max_iter=1000).fit(factor_x)
factor_rows = [0, 37, 99]
lda_x = np.array([[8, 5, 1, 0, 0, 0], [7, 6, 0, 0, 0, 1], [9, 4, 1, 0, 0, 0], [6, 7, 0, 1, 0, 0], [0, 0, 0, 8, 5, 1], [0, 1, 0, 7, 6, 0], [1, 0, 0, 9, 4, 1], [0, 0, 1, 6, 7, 0]], dtype=float)
lda = LatentDirichletAllocation(n_components=2, learning_method='batch', max_iter=30, random_state=0, evaluate_every=-1).fit(lda_x)

pls_x = np.array([[i / 5, np.sin(i), (i % 3) - 1] for i in range(20)], dtype=float)
pls_y = np.column_stack((2 * pls_x[:, 0] - pls_x[:, 1] + .5 * pls_x[:, 2], -pls_x[:, 0] + .3 * pls_x[:, 1] + 2 * pls_x[:, 2]))
pls = PLSRegression(n_components=2, scale=True, max_iter=500, tol=1e-10).fit(pls_x, pls_y)
cca = CCA(n_components=2, scale=True, max_iter=1000, tol=1e-10).fit(pls_x, pls_y)
pls_rows = [0, 7, 19]

linear_x = np.array([[i / 10, np.sin(i), (i % 4) - 1.5, np.cos(i / 3)] for i in range(30)], dtype=float)
linear_y = 1.5 + 2 * linear_x[:, 0] - .7 * linear_x[:, 1] + .3 * linear_x[:, 2] + .05 * np.sin(np.arange(30) * 1.7)
bayesian = BayesianRidge(max_iter=500, tol=1e-8).fit(linear_x, linear_y)
ard = ARDRegression(max_iter=500, tol=1e-8, threshold_lambda=1e4).fit(linear_x, linear_y)
linear_rows = [0, 11, 29]

robust_x = np.array([[i / 5, (i % 3) - 1] for i in range(30)], dtype=float)
robust_y = 2 + 1.7 * robust_x[:, 0] - .8 * robust_x[:, 1] + .03 * np.sin(np.arange(30))
robust_y[[3, 17, 25]] += np.array([8, -10, 7])
huber = HuberRegressor(epsilon=1.35, alpha=.0001, max_iter=500, tol=1e-8).fit(robust_x, robust_y)
ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=3, residual_threshold=.5, max_trials=100, random_state=0).fit(robust_x, robust_y)
theil = TheilSenRegressor(max_subpopulation=10000, max_iter=500, tol=1e-6, random_state=0).fit(robust_x[:12], robust_y[:12])
quantile = QuantileRegressor(quantile=.5, alpha=0, fit_intercept=True, solver='highs').fit(robust_x, robust_y)
quantile_regularized = QuantileRegressor(quantile=.5, alpha=.1, fit_intercept=True, solver='highs').fit(robust_x, robust_y)
robust_rows = [0, 11, 29]

glm_x = np.array([[i / 10, (i % 5) - 2] for i in range(1, 31)], dtype=float)
poisson_y = np.round(np.exp(.3 + .25 * glm_x[:, 0] - .12 * glm_x[:, 1])).astype(float)
gamma_y = np.exp(.5 + .2 * glm_x[:, 0] + .08 * glm_x[:, 1]) * (1 + .05 * np.sin(np.arange(30)))
poisson = PoissonRegressor(alpha=.1, max_iter=500, tol=1e-10).fit(glm_x, poisson_y)
gamma = GammaRegressor(alpha=.1, max_iter=500, tol=1e-10).fit(glm_x, gamma_y)
tweedie = TweedieRegressor(power=1.5, alpha=.1, link='log', max_iter=500, tol=1e-10).fit(glm_x, gamma_y)
tweedie_identity = TweedieRegressor(power=1.5, alpha=.1, link='identity', max_iter=500, tol=1e-10).fit(glm_x, gamma_y)
glm_rows = [0, 14, 29]

advanced_x = np.array([[0.], [1.], [2.], [3.]])
spline_query = np.array([[0.], [.5], [2.], [3.], [4.]])
spline = SplineTransformer(n_knots=3, degree=2, include_bias=True).fit(advanced_x)
spline_periodic = SplineTransformer(n_knots=5, degree=3, extrapolation='periodic').fit(advanced_x)
spline_continue = SplineTransformer(n_knots=3, degree=2, extrapolation='continue').fit(advanced_x)
spline_linear = SplineTransformer(n_knots=3, degree=2, extrapolation='linear').fit(advanced_x)
spline_far_query = np.array([[-5.], [4.]])
spline_explicit_x = np.array([[0., 0.], [1., 2.], [3., 4.]])
spline_explicit_knots = np.array([[0., 0.], [1., 2.], [3., 4.]])
spline_explicit = SplineTransformer(knots=spline_explicit_knots, degree=2).fit(spline_explicit_x)
target_x = np.array([['a'], ['a'], ['b'], ['b'], ['c'], ['c']], dtype=object)
target_y = np.array([0., 2.1, 10.2, 12.3, 20.4, 22.5])
target_encoder = TargetEncoder(smooth=2., cv=KFold(n_splits=3, shuffle=False), target_type='continuous')
target_cross_fit = target_encoder.fit_transform(target_x, target_y)
target_query = np.array([['a'], ['b'], ['c'], ['unknown']], dtype=object)
target_auto = TargetEncoder(smooth='auto', cv=KFold(n_splits=3, shuffle=False), target_type='continuous').fit(target_x, target_y)
binary_x = np.array([['a'], ['a'], ['b'], ['b'], ['c'], ['c'], ['a'], ['a'], ['b'], ['b'], ['c'], ['c']], dtype=object)
binary_y = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
binary_encoder = TargetEncoder(smooth=2., cv=StratifiedKFold(n_splits=2, shuffle=False), target_type='binary')
binary_cross_fit = binary_encoder.fit_transform(binary_x, binary_y)
multiclass_y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
multiclass_encoder = TargetEncoder(smooth=2., cv=StratifiedKFold(n_splits=2, shuffle=False), target_type='multiclass')
multiclass_cross_fit = multiclass_encoder.fit_transform(binary_x, multiclass_y)
imbalanced_x = np.array([['a'], ['b'], ['a'], ['c'], ['b'], ['a'], ['b'], ['c'], ['c']], dtype=object)
imbalanced_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
imbalanced_encoder = TargetEncoder(smooth=2., cv=3, shuffle=False, target_type='binary')
imbalanced_cross_fit = imbalanced_encoder.fit_transform(imbalanced_x, imbalanced_y)
first_order_x = np.array([[value] for value in ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']], dtype=object)
first_order_y = np.array([2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1])
first_order_encoder = TargetEncoder(smooth=2., cv=3, shuffle=False, target_type='multiclass')
first_order_cross_fit = first_order_encoder.fit_transform(first_order_x, first_order_y)
single_class_y = np.zeros(len(target_x), dtype=int)
single_class_encoder = TargetEncoder(smooth=2., cv=3, shuffle=False, target_type='auto')
single_class_cross_fit = single_class_encoder.fit_transform(target_x, single_class_y)
multilabel_y = [['sci-fi', 'thriller'], ['comedy'], []]
multilabel = MultiLabelBinarizer().fit(multilabel_y)
neighbors_x = np.array([[0.], [2.], [5.]])
neighbors_query = np.array([[1.], [4.]])
nearest = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(neighbors_x)
neighbor_distances, neighbor_indices = nearest.kneighbors(neighbors_query)
own_distances, own_indices = nearest.kneighbors()
radius_distances, radius_indices = nearest.radius_neighbors(neighbors_query, radius=1.5)
self_radius_distances, self_radius_indices = nearest.radius_neighbors(radius=0)
ttr = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
).fit(np.array([[0.], [1.], [2.]]), np.array([1., 3., 7.]))
inspection_x = np.column_stack((np.arange(20, dtype=float), np.arange(20) % 2))
inspection_y = 3 * inspection_x[:, 0] + 2
inspection_model = LinearRegression().fit(inspection_x, inspection_y)
permutation = permutation_importance(inspection_model, inspection_x, inspection_y, n_repeats=4, random_state=7)
dependence = partial_dependence(inspection_model, inspection_x, [0], grid_resolution=4, percentiles=(0, 1))

fixture = {
    'kernel_ridge': {
        'X': kernel_x.tolist(), 'y': kernel_y.tolist(), 'query': kernel_query.tolist(),
        'prediction': kernel_ridge.predict(kernel_query).tolist(), 'dual_coef': kernel_ridge.dual_coef_.tolist(),
    },
    'kernel_density': {
        'X': density_x.tolist(), 'query': density_query.tolist(),
        'gaussian': kde_gaussian.score_samples(density_query).tolist(),
        'epanechnikov': kde_epanechnikov.score_samples(density_query).tolist(),
        'cosine_dimensions': cosine_dimensions, 'cosine_origin_scores': cosine_origin_scores,
    },
    'covariance': {
        'X': covariance_x.tolist(),
        'empirical': {'location': empirical.location_.tolist(), 'covariance': empirical.covariance_.tolist(), 'precision': empirical.precision_.tolist()},
        'shrunk': {'covariance': shrunk.covariance_.tolist()},
        'ledoit_wolf': {'covariance': ledoit.covariance_.tolist(), 'shrinkage': float(ledoit.shrinkage_)},
        'oas': {'covariance': oas.covariance_.tolist(), 'shrinkage': float(oas.shrinkage_)},
        'graphical_lasso': {'covariance': graphical.covariance_.tolist(), 'precision': graphical.precision_.tolist()},
    },
    'random_projection': {
        'jl_min_dim': int(johnson_lindenstrauss_min_dim(100, eps=.5)),
        'geometry_X': np.array([[np.sin(i * .7 + j * .13) + np.cos(i * .11 - j * .19) for j in range(60)] for i in range(40)]).tolist(),
        'geometry_components': 20,
    },
    'hashing_vectorizer': {'documents': documents, 'matrix': hashing_matrix.tolist(), 'default_matrix': hashing_default.tolist()},
    'dict_vectorizer': {
        'rows': dict_rows, 'feature_names': dict_vectorizer.get_feature_names_out().tolist(),
        'matrix': dict_vectorizer.transform(dict_rows).tolist(),
    },
    'feature_hasher': {'rows': hasher_rows, 'matrix': feature_hasher.transform(hasher_rows).toarray().tolist()},
    'clustering': {
        'X': cluster_x.tolist(),
        'birch': {
            'subcluster_centers': birch.subcluster_centers_.tolist(),
            'subcluster_labels': birch.subcluster_labels_.tolist(), 'labels': birch.labels_.tolist(),
        },
        'birch_tree': {
            'X': birch_tree_x.tolist(), 'labels': birch_tree.labels_.tolist(),
            'subcluster_centers': birch_tree.subcluster_centers_.tolist(), 'subcluster_labels': birch_tree.subcluster_labels_.tolist(),
        },
        'affinity_propagation': {
            'center_indices': affinity.cluster_centers_indices_.tolist(), 'centers': affinity.cluster_centers_.tolist(),
            'labels': affinity.labels_.tolist(), 'n_iter': int(affinity.n_iter_),
        },
        'bisecting_kmeans': {
            'centers': bisecting.cluster_centers_.tolist(), 'labels': bisecting.labels_.tolist(),
            'inertia': float(bisecting.inertia_),
        },
    },
    'decomposition': {
        'factor_analysis': {
            'X': factor_x.tolist(), 'rows': factor_rows, 'mean': factor.mean_.tolist(),
            'noise_variance': factor.noise_variance_.tolist(), 'covariance': factor.get_covariance().tolist(),
            'transform': factor.transform(factor_x[factor_rows]).tolist(), 'n_iter': int(factor.n_iter_),
            'varimax_components': factor_varimax.components_.tolist(), 'varimax_transform': factor_varimax.transform(factor_x[factor_rows]).tolist(),
        },
        'lda': {
            'X': lda_x.tolist(),
            'topics': (lda.components_ / lda.components_.sum(axis=1, keepdims=True)).tolist(),
            'transform': lda.transform(lda_x).tolist(),
        },
    },
    'cross_decomposition': {
        'X': pls_x.tolist(), 'Y': pls_y.tolist(), 'rows': pls_rows,
        'pls': {
            'x_weights': pls.x_weights_.tolist(), 'x_loadings': pls.x_loadings_.tolist(),
            'transform': pls.transform(pls_x[pls_rows]).tolist(), 'prediction': pls.predict(pls_x[pls_rows]).tolist(), 'score': float(pls.score(pls_x, pls_y)),
        },
        'cca': {
            'x_weights': cca.x_weights_.tolist(), 'x_loadings': cca.x_loadings_.tolist(),
            'transform': cca.transform(pls_x[pls_rows]).tolist(), 'prediction': cca.predict(pls_x[pls_rows]).tolist(), 'score': float(cca.score(pls_x, pls_y)),
        },
    },
    'linear_models': {
        'X': linear_x.tolist(), 'y': linear_y.tolist(), 'rows': linear_rows,
        'bayesian_ridge': {
            'coef': bayesian.coef_.tolist(), 'intercept': float(bayesian.intercept_), 'alpha': float(bayesian.alpha_),
            'lambda': float(bayesian.lambda_), 'prediction': bayesian.predict(linear_x[linear_rows]).tolist(),
            'std': bayesian.predict(linear_x[linear_rows], return_std=True)[1].tolist(),
        },
        'ard': {
            'coef': ard.coef_.tolist(), 'intercept': float(ard.intercept_), 'alpha': float(ard.alpha_),
            'lambda': ard.lambda_.tolist(), 'prediction': ard.predict(linear_x[linear_rows]).tolist(),
        },
        'robust': {
            'X': robust_x.tolist(), 'y': robust_y.tolist(), 'rows': robust_rows,
            'huber': {'coef': huber.coef_.tolist(), 'intercept': float(huber.intercept_), 'scale': float(huber.scale_), 'outliers': huber.outliers_.tolist()},
            'ransac': {'coef': ransac.estimator_.coef_.tolist(), 'intercept': float(ransac.estimator_.intercept_), 'inliers': ransac.inlier_mask_.tolist(), 'prediction': ransac.predict(robust_x[robust_rows]).tolist()},
            'theil_sen': {'coef': theil.coef_.tolist(), 'intercept': float(theil.intercept_), 'prediction': theil.predict(robust_x[robust_rows]).tolist()},
            'quantile': {'coef': quantile.coef_.tolist(), 'intercept': float(quantile.intercept_), 'prediction': quantile.predict(robust_x[robust_rows]).tolist()},
            'quantile_regularized': {'coef': quantile_regularized.coef_.tolist(), 'intercept': float(quantile_regularized.intercept_), 'prediction': quantile_regularized.predict(robust_x[robust_rows]).tolist()},
        },
        'glm': {
            'X': glm_x.tolist(), 'poisson_y': poisson_y.tolist(), 'gamma_y': gamma_y.tolist(), 'rows': glm_rows,
            'poisson': {'coef': poisson.coef_.tolist(), 'intercept': float(poisson.intercept_), 'prediction': poisson.predict(glm_x[glm_rows]).tolist()},
            'gamma': {'coef': gamma.coef_.tolist(), 'intercept': float(gamma.intercept_), 'prediction': gamma.predict(glm_x[glm_rows]).tolist()},
            'tweedie': {'coef': tweedie.coef_.tolist(), 'intercept': float(tweedie.intercept_), 'prediction': tweedie.predict(glm_x[glm_rows]).tolist()},
            'tweedie_identity': {'coef': tweedie_identity.coef_.tolist(), 'intercept': float(tweedie_identity.intercept_), 'prediction': tweedie_identity.predict(glm_x[glm_rows]).tolist()},
        },
    },
    'advanced': {
        'spline': {
            'X': advanced_x.tolist(), 'query': spline_query.tolist(), 'transform': spline.transform(spline_query).tolist(),
            'far_query': spline_far_query.tolist(), 'continue': spline_continue.transform(spline_far_query).tolist(),
            'linear': spline_linear.transform(spline_far_query).tolist(),
            'periodic': spline_periodic.transform(spline_far_query).tolist(),
            'explicit_X': spline_explicit_x.tolist(), 'explicit_knots': spline_explicit_knots.tolist(),
            'explicit': spline_explicit.transform(spline_explicit_x).tolist(),
        },
        'target_encoder': {
            'X': target_x.tolist(), 'y': target_y.tolist(), 'query': target_query.tolist(),
            'cross_fit': target_cross_fit.tolist(), 'transform': target_encoder.transform(target_query).tolist(),
            'auto': target_auto.transform(target_query).tolist(),
            'binary_X': binary_x.tolist(), 'binary_y': binary_y.tolist(), 'binary_cross_fit': binary_cross_fit.tolist(),
            'binary_transform': binary_encoder.transform(target_query).tolist(),
            'multiclass_y': multiclass_y.tolist(), 'multiclass_cross_fit': multiclass_cross_fit.tolist(),
            'multiclass_transform': multiclass_encoder.transform(target_query).tolist(),
            'imbalanced_X': imbalanced_x.tolist(), 'imbalanced_y': imbalanced_y.tolist(), 'imbalanced_cross_fit': imbalanced_cross_fit.tolist(),
            'first_order_X': first_order_x.tolist(), 'first_order_y': first_order_y.tolist(), 'first_order_cross_fit': first_order_cross_fit.tolist(),
            'single_class_y': single_class_y.tolist(), 'single_class_cross_fit': single_class_cross_fit.tolist(),
        },
        'multi_label_binarizer': {
            'y': multilabel_y, 'classes': multilabel.classes_.tolist(), 'transform': multilabel.transform(multilabel_y).tolist(),
        },
        'nearest_neighbors': {
            'X': neighbors_x.tolist(), 'query': neighbors_query.tolist(),
            'distances': neighbor_distances.tolist(), 'indices': neighbor_indices.tolist(),
            'own_distances': own_distances.tolist(), 'own_indices': own_indices.tolist(),
            'radius_distances': [row.tolist() for row in radius_distances], 'radius_indices': [row.tolist() for row in radius_indices],
            'self_radius_distances': [row.tolist() for row in self_radius_distances], 'self_radius_indices': [row.tolist() for row in self_radius_indices],
        },
        'transformed_target': {'prediction': ttr.predict(np.array([[3.]])).tolist()},
        'inspection': {
            'X': inspection_x.tolist(), 'y': inspection_y.tolist(),
            'permutation_mean': permutation.importances_mean.tolist(),
            'partial_grid': [row.tolist() for row in dependence['grid_values']],
            'partial_average': dependence['average'].tolist(),
        },
    },
}

target = Path(__file__).resolve().parent.parent / 'test_data' / 'wave_c.json'
target.write_text(json.dumps(fixture, separators=(',', ':')))
