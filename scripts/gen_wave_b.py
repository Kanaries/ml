"""Generate deterministic sklearn reference data for the Phase 3 Wave B APIs."""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA, IncrementalPCA, KernelPCA, NMF
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import ClassifierChain, RegressorChain
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import SelfTrainingClassifier


def serializable_matrix(matrix):
    return [[None if np.isnan(value) else float(value) for value in row] for row in np.asarray(matrix)]


documents = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
count = CountVectorizer().fit(documents)
count_options = CountVectorizer(stop_words=['the'], ngram_range=(1, 2), min_df=2, binary=True).fit(documents)
counts = count.transform(documents)
tfidf = TfidfTransformer().fit(counts)

kernel_x = np.array([[1, 2], [2, 4.1], [3, 5.9], [4, 8.2], [5, 9.8]], dtype=float)
kernel = KernelPCA(n_components=2, kernel='linear', eigen_solver='dense', random_state=0).fit(kernel_x)
sigmoid = KernelPCA(n_components=3, kernel='sigmoid', gamma=1, coef0=1, eigen_solver='dense').fit(
    np.array([[-2], [-1], [0], [1], [2]], dtype=float)
)

ica_x = np.array([[0], [1], [3], [6]], dtype=float)
ica_unit = FastICA(n_components=1, whiten='unit-variance', random_state=0, max_iter=1000, tol=1e-8).fit(ica_x)
ica_arbitrary = FastICA(n_components=1, whiten='arbitrary-variance', random_state=0, max_iter=1000, tol=1e-8).fit(ica_x)
ica_time = np.arange(200, dtype=float) / 20
ica_sources = np.column_stack((np.sin(ica_time), np.where(np.arange(200) % 40 < 20, 1.0, -1.0)))
ica_deflation_x = ica_sources @ np.array([[1.0, .3], [.5, 1.0]])
ica_deflation = FastICA(n_components=2, algorithm='deflation', whiten='unit-variance', random_state=0, max_iter=1000, tol=1e-8).fit(ica_deflation_x)

nmf_x = np.array([[1, 2, 0], [.5, 1.5, 1.5], [0, 1, 3], [1, 3, 3]], dtype=float)
nmf = NMF(n_components=2, init='nndsvda', solver='mu', beta_loss='frobenius', max_iter=1000, tol=1e-7, random_state=0)
nmf_w = nmf.fit_transform(nmf_x)
nmf_eps_x = np.array([[1.0, 0.0], [0.0, 1e-14]])
nmf_eps = NMF(n_components=2, init='nndsvda', solver='mu', beta_loss='frobenius', max_iter=20, tol=0, random_state=0)
nmf_eps_w = nmf_eps.fit_transform(nmf_eps_x)

ipca_x = np.array([[i / 10, np.sin(i), (i % 5) - 2] for i in range(50)], dtype=float)
ipca_partial = IncrementalPCA(n_components=2)
for start, end in [(0, 17), (17, 33), (33, 50)]:
    ipca_partial.partial_fit(ipca_x[start:end])
ipca_fit = IncrementalPCA(n_components=2).fit(ipca_x)
ipca_merged_remainder_x = ipca_x[:16]
ipca_merged_remainder = IncrementalPCA(n_components=2).fit(ipca_merged_remainder_x)
ipca_rows = [0, 10, 49]

imputer_x = np.array([[0, 1], [1, 3], [2, np.nan], [3, 7], [4, np.nan], [5, 11]], dtype=float)
imputer = IterativeImputer(estimator=Ridge(alpha=1e-6), max_iter=20, tol=1e-8).fit(imputer_x)
small_rng = np.random.RandomState(2)
small_imputer_x = small_rng.rand(30, 4) * .01
small_imputer_x[:, 1] = .6 * small_imputer_x[:, 0] + .3 * small_imputer_x[:, 2] + .05 * small_rng.rand(30) * .01
small_imputer_x[small_rng.rand(*small_imputer_x.shape) < .22] = np.nan
small_imputer = IterativeImputer(estimator=Ridge(alpha=1e-6), max_iter=20, tol=.03, random_state=0).fit(small_imputer_x)
zero_scale_imputer_x = np.array([[0, 0], [0, np.nan], [0, 0]], dtype=float)
zero_scale_imputer = IterativeImputer(estimator=Ridge(alpha=1e-6), max_iter=3, tol=1e-3).fit(zero_scale_imputer_x)
norm_rng = np.random.RandomState(0)
norm_imputer_x = norm_rng.rand(20, 4)
norm_imputer_x[norm_rng.rand(*norm_imputer_x.shape) < .35] = np.nan
norm_imputer = IterativeImputer(estimator=Ridge(alpha=1e-6), max_iter=20, tol=.05, random_state=0).fit(norm_imputer_x)
descending_imputer_x = np.array([
    [np.nan, 1, 2], [1, np.nan, 4], [2, 4, np.nan], [3, 6, 8], [4, 8, 10], [5, 10, 12],
], dtype=float)
descending_imputer = IterativeImputer(
    estimator=Ridge(alpha=1e-6), max_iter=2, tol=0, imputation_order='descending'
).fit(descending_imputer_x)

semi_x = np.array([[-3], [-2], [-1.8], [-1.5], [1.5], [1.8], [2], [3]], dtype=float)
semi_y = np.array([0, 0, -1, -1, -1, -1, 1, 1])
self_training = SelfTrainingClassifier(estimator=GaussianNB(), threshold=.8, max_iter=10).fit(semi_x, semi_y)
no_change_x = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
no_change_y = np.array([0, 0, -1, 1, 1])
no_change_self_training = SelfTrainingClassifier(estimator=GaussianNB(), threshold=.99, max_iter=1).fit(no_change_x, no_change_y)

chain_x = np.array([[i - 10, i % 3] for i in range(20)], dtype=float)
chain_y = np.array([[int(row[0] >= 0), int(row[0] >= 0 and row[1] > 0)] for row in chain_x])
classifier_chain = ClassifierChain(estimator=GaussianNB()).fit(chain_x, chain_y)
chain_rows = [0, 9, 10, 19]

regression_x = np.array([[i / 3, (i % 4) - 2] for i in range(20)], dtype=float)
regression_y = np.array([[2 * row[0] - row[1] + 1, -row[0] + 3 * row[1] - 2] for row in regression_x])
regressor_chain = RegressorChain(estimator=Ridge(alpha=1e-10), order=[1, 0]).fit(regression_x, regression_y)
regression_rows = [0, 7, 19]

split_groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6])
split_y = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
split_x = np.arange(len(split_y)).reshape(-1, 1)
split_tests = [test.tolist() for _, test in StratifiedGroupKFold(n_splits=3).split(split_x, split_y, split_groups)]
near_tie_groups = np.array([0, 0, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9, 10, 11, 11, 11, 11])
near_tie_y = np.array([2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 1, 2, 1, 0, 0, 2, 2, 1, 2, 1, 2, 1, 0, 1, 1])
near_tie_x = np.arange(len(near_tie_y)).reshape(-1, 1)
near_tie_tests = [test.tolist() for _, test in StratifiedGroupKFold(n_splits=3).split(near_tie_x, near_tie_y, near_tie_groups)]

fixture = {
    'text': {
        'documents': documents,
        'count_feature_names': count.get_feature_names_out().tolist(),
        'count_matrix': counts.toarray().tolist(),
        'options_feature_names': count_options.get_feature_names_out().tolist(),
        'options_matrix': count_options.transform(documents).toarray().tolist(),
        'idf': tfidf.idf_.tolist(),
        'tfidf_matrix': tfidf.transform(counts).toarray().tolist(),
    },
    'kernel_pca': {
        'X': kernel_x.tolist(),
        'transform': kernel.transform(kernel_x).tolist(),
        'eigenvalues': kernel.eigenvalues_.tolist(),
        'sigmoid_eigenvalues': sigmoid.eigenvalues_.tolist(),
    },
    'fast_ica': {
        'X': ica_x.tolist(),
        'unit': ica_unit.transform(ica_x).tolist(),
        'arbitrary': ica_arbitrary.transform(ica_x).tolist(),
        'deflation_X': ica_deflation_x.tolist(),
        'deflation': ica_deflation.transform(ica_deflation_x).tolist(),
    },
    'nmf': {
        'X': nmf_x.tolist(), 'W': nmf_w.tolist(), 'H': nmf.components_.tolist(),
        'reconstruction_error': float(nmf.reconstruction_err_), 'n_iter': int(nmf.n_iter_),
        'eps_truncation': {
            'X': nmf_eps_x.tolist(), 'W': nmf_eps_w.tolist(), 'H': nmf_eps.components_.tolist(),
            'reconstruction_error': float(nmf_eps.reconstruction_err_),
        },
    },
    'incremental_pca': {
        'X': ipca_x.tolist(), 'rows': ipca_rows,
        'partial': {
            'mean': ipca_partial.mean_.tolist(), 'explained_variance': ipca_partial.explained_variance_.tolist(),
            'singular_values': ipca_partial.singular_values_.tolist(),
            'transform': ipca_partial.transform(ipca_x[ipca_rows]).tolist(),
        },
        'fit': {
            'mean': ipca_fit.mean_.tolist(), 'explained_variance': ipca_fit.explained_variance_.tolist(),
            'singular_values': ipca_fit.singular_values_.tolist(),
            'transform': ipca_fit.transform(ipca_x[ipca_rows]).tolist(),
        },
        'merged_remainder': {
            'X': ipca_merged_remainder_x.tolist(),
            'explained_variance': ipca_merged_remainder.explained_variance_.tolist(),
            'singular_values': ipca_merged_remainder.singular_values_.tolist(),
        },
    },
    'iterative_imputer': {
        'X': serializable_matrix(imputer_x), 'transform': imputer.transform(imputer_x).tolist(),
        'future': imputer.transform([[6, np.nan]]).tolist(), 'n_iter': int(imputer.n_iter_),
        'small_scale': {
            'X': serializable_matrix(small_imputer_x), 'transform': small_imputer.transform(small_imputer_x).tolist(),
            'n_iter': int(small_imputer.n_iter_),
        },
        'zero_scale': {
            'X': serializable_matrix(zero_scale_imputer_x),
            'transform': zero_scale_imputer.transform(zero_scale_imputer_x).tolist(),
            'n_iter': int(zero_scale_imputer.n_iter_),
        },
        'infinity_norm': {
            'X': serializable_matrix(norm_imputer_x), 'transform': norm_imputer.transform(norm_imputer_x).tolist(),
            'n_iter': int(norm_imputer.n_iter_),
        },
        'descending_ties': {
            'X': serializable_matrix(descending_imputer_x),
            'transform': descending_imputer.transform(descending_imputer_x).tolist(),
            'n_iter': int(descending_imputer.n_iter_),
        },
    },
    'self_training': {
        'X': semi_x.tolist(), 'y': semi_y.tolist(), 'transduction': self_training.transduction_.tolist(),
        'labeled_iteration': self_training.labeled_iter_.tolist(), 'n_iter': int(self_training.n_iter_),
        'termination': self_training.termination_condition_,
        'max_iter_no_change': {
            'X': no_change_x.tolist(), 'y': no_change_y.tolist(),
            'n_iter': int(no_change_self_training.n_iter_),
            'termination': no_change_self_training.termination_condition_,
        },
    },
    'classifier_chain': {
        'X': chain_x.tolist(), 'Y': chain_y.tolist(), 'prediction': classifier_chain.predict(chain_x).astype(int).tolist(),
        'rows': chain_rows, 'probability': classifier_chain.predict_proba(chain_x[chain_rows]).tolist(),
    },
    'regressor_chain': {
        'X': regression_x.tolist(), 'Y': regression_y.tolist(), 'rows': regression_rows,
        'prediction': regressor_chain.predict(regression_x[regression_rows]).tolist(),
    },
    'stratified_group_kfold': {
        'groups': split_groups.tolist(), 'y': split_y.tolist(), 'test_indices': split_tests,
        'near_tie': {
            'groups': near_tie_groups.tolist(), 'y': near_tie_y.tolist(), 'test_indices': near_tie_tests,
        },
    },
}

target = Path(__file__).resolve().parent.parent / 'test_data' / 'wave_b.json'
target.write_text(json.dumps(fixture, separators=(',', ':')))
