import numpy as np
from sklearn.neighbors import BallTree
import json, os

rng = np.random.RandomState(0)
X = rng.random_sample((20, 3))
leaf_size = 2
k = 3
test = rng.random_sample((5, 3))
radius = 0.3
tree = BallTree(X, leaf_size=leaf_size)
dist, ind = tree.query(test, k=k)
ind_r, dist_r = tree.query_radius(test, r=radius, return_distance=True)

os.makedirs('test_data', exist_ok=True)
with open('test_data/balltree.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'leaf_size': leaf_size,
        'test': test.tolist(),
        'k': k,
        'radius': radius,
        'query_indices': ind.tolist(),
        'query_distances': dist.tolist(),
        'radius_indices': [i.tolist() for i in ind_r],
        'radius_distances': [d.tolist() for d in dist_r]
    }, f)
