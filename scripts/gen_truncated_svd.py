import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import json, os

np.random.seed(0)
X_dense = np.random.rand(100, 100)
X_dense[:, 2 * np.arange(50)] = 0
X = csr_matrix(X_dense)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)
X_test_dense = np.random.rand(10, 100)
X_test_dense[:, 2 * np.arange(50)] = 0
X_test = csr_matrix(X_test_dense)
trans = svd.transform(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/truncated_svd.json', 'w') as f:
    json.dump({
        'X': X_dense.tolist(),
        'X_test': X_test_dense.tolist(),
        'expected': trans.tolist(),
        'components': svd.components_.tolist(),
        'explained_variance': svd.explained_variance_.tolist(),
        'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
        'singular_values': svd.singular_values_.tolist()
    }, f)
