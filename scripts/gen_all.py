import subprocess, os, sys

here = os.path.dirname(os.path.abspath(__file__))

def run(script):
    subprocess.run([sys.executable, os.path.join(here, script)], check=True)

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
    'gen_label_spreading.py'
]

for s in scripts:
    run(s)
