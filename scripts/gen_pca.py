import numpy as np
from sklearn.decomposition import PCA
import json, os

np.random.seed(0)
X = np.random.randn(50, 3)
pca = PCA(n_components=2)
pca.fit(X)
X_test = np.random.randn(10, 3)
trans = pca.transform(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/pca.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'X_test': X_test.tolist(),
        'expected': trans.tolist(),
        'components': pca.components_.tolist(),
        'mean': pca.mean_.tolist(),
        'explained_variance': pca.explained_variance_.tolist()
    }, f)
