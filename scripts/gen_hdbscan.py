import numpy as np
from sklearn.cluster import HDBSCAN
import json, os

# Varying-density dataset: two tight clusters, one sparse cluster and
# background noise placed in a strip far from every cluster. A single
# fixed-eps DBSCAN cannot recover all three clusters at once, which is
# exactly the scenario HDBSCAN handles.
#
# Platform-stable construction: rng.uniform is pure bit arithmetic (randn
# goes through libm log/sqrt whose last ULP differs between glibc and Apple
# libm), and the noise strip keeps points far from cluster boundaries so no
# marginal point can flip between cluster and noise across platforms.
rng = np.random.RandomState(42)
A = rng.uniform(-0.5, 0.5, size=(30, 2)) + np.array([0, 0])
# B sits between min_cluster_size 8 and 25: a cluster at mcs=8, dissolved at mcs=25
B = rng.uniform(-0.5, 0.5, size=(15, 2)) + np.array([4, 0])
C = rng.uniform(-3.0, 3.0, size=(40, 2)) + np.array([15, 15])
noise = np.column_stack([
    rng.uniform(-8, -5, size=10),
    rng.uniform(-8, 25, size=10),
])
X = np.vstack([A, B, C, noise])

labels_mcs8 = HDBSCAN(min_cluster_size=8).fit_predict(X)
n_clusters_8 = len(set(labels_mcs8)) - (1 if -1 in labels_mcs8 else 0)
n_noise_8 = int(np.sum(labels_mcs8 == -1))
assert n_clusters_8 == 3, f'expected 3 clusters, got {n_clusters_8}'
assert n_noise_8 > 0, 'expected at least one noise point'

labels_mcs25 = HDBSCAN(min_cluster_size=25).fit_predict(X)
n_clusters_25 = len(set(labels_mcs25)) - (1 if -1 in labels_mcs25 else 0)
assert n_clusters_25 < n_clusters_8, 'min_cluster_size=25 must dissolve the 15-point cluster'

labels_eps2 = HDBSCAN(min_cluster_size=8, cluster_selection_epsilon=2.0).fit_predict(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/hdbscan.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'expected_mcs8': labels_mcs8.tolist(),
        'expected_mcs25': labels_mcs25.tolist(),
        'expected_eps2': labels_eps2.tolist()
    }, f)
