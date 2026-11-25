import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent

def run(script):
    subprocess.run([sys.executable, here / script], cwd=here.parent, check=True)

scripts = [
    'gen_linear_regression.py',
    'gen_decision_tree_classifier.py',
    'gen_extra_tree_classifier.py',
    'gen_decision_tree_regressor.py',
    'gen_ada_boost_regressor.py',
    'gen_extra_tree_regressor.py',
    'gen_isolation_forest.py',
    'gen_knn.py',
    'gen_ball_tree.py',
    'gen_kd_tree.py',
    'gen_kmeans.py',
    'gen_e_means.py',
    'gen_mean_shift.py',
    'gen_dbscan.py',
    'gen_hdbscan.py',
    'gen_optics.py',
    'gen_logistic_regression.py',
    'gen_linear_svr.py',
    'gen_elastic_net.py',
    'gen_ridge_classifier.py',
    'gen_adaboost_classifier.py',
    'gen_bernoulli_nb.py',
    'gen_categorical_nb.py',
    'gen_gaussian_nb.py',
    'gen_gaussian_process.py',
    'gen_multinomial_nb.py',
    'gen_complement_nb.py',
    'gen_svc.py',
    'gen_nu_svc.py',
    'gen_bagging_classifier.py',
    'gen_kneighbors_regressor.py',
    'gen_radius_neighbors_classifier.py',
    'gen_radius_neighbors_regressor.py',
    'gen_nearest_centroid.py',
    'gen_random_forest_classifier.py',
    'gen_random_forest_regressor.py',
    'gen_gradient_boosting_regressor.py',
    'gen_gradient_boosting_classifier.py',
    'gen_hist_gradient_boosting.py',
    'gen_xgboost_regressor.py',
    'gen_xgboost_classifier.py',
    'gen_gradient_boosting_classifier_multi.py',
    'gen_xgboost_classifier_multi.py',
    'gen_adaboost_classifier_multi.py',
    'gen_bernoulli_rbm.py',
    'gen_pca.py',
    'gen_truncated_svd.py',
    'gen_sparse_pca.py',
    'gen_spectral_embedding.py',
    'gen_mds.py',
    'gen_lle.py',
    'gen_tsne.py',
    'gen_label_propagation.py',
    'gen_label_spreading.py',
    'gen_wave_b.py',
    'gen_wave_c.py',
    'gen_kernel_approximation.py',
    'gen_phase3_e2e.py'
]

def fixture_versions(directory):
    versions = {}
    for path in directory.rglob('*'):
        if path.name.endswith(('.json', '.json.gz')) and path.is_file():
            stat = path.stat()
            versions[path.relative_to(directory).as_posix()] = (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)
    return versions


if __name__ == '__main__':
    directory = here.parent / 'test_data'
    directory.mkdir(exist_ok=True)
    before = fixture_versions(directory)
    for script in scripts:
        run(script)
    after = fixture_versions(directory)
    # A deterministic generator can write exactly the same bytes again.
    written = sorted(name for name, version in after.items() if before.get(name) != version)
    reused = sorted(after.keys() - set(written))
    print(f'Regenerated ({len(written)}): {", ".join(written) or "none"}')
    print(f'Reused existing fixtures ({len(reused)}): {", ".join(reused) or "none"}')
