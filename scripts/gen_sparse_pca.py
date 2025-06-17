import numpy as np
from sklearn.decomposition import SparsePCA
import json, os

np.random.seed(0)
X = np.random.randn(50, 10)
spca = SparsePCA(n_components=5, alpha=0, random_state=0)
spca.fit(X)
X_test = np.random.randn(10, 10)
trans = spca.transform(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/sparse_pca.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'X_test': X_test.tolist(),
        'expected': trans.tolist()
    }, f)
