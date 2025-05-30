import subprocess, os

here = os.path.dirname(os.path.abspath(__file__))

def run(script):
    subprocess.run(['python', os.path.join(here, script)], check=True)

scripts = [
    'gen_linear_regression.py',
    'gen_decision_tree_classifier.py',
    'gen_decision_tree_regressor.py',
    'gen_isolation_forest.py',
    'gen_knn.py',
    'gen_kmeans.py',
    'gen_mean_shift.py',
    'gen_dbscan.py',
    'gen_optics.py',
    'gen_logistic_regression.py',
    'gen_svc.py',
    'gen_pca.py',
    'gen_mds.py'
]

for s in scripts:
    run(s)
