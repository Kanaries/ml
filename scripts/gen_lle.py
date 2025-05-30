import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import json, os

np.random.seed(0)
X = np.random.randn(40, 5)
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2, random_state=0)
lle.fit(X)
X_test = np.random.randn(10, 5)
trans = lle.transform(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/lle.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'X_test': X_test.tolist(),
        'expected': trans.tolist()
    }, f)
