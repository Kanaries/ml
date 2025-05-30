import numpy as np
from sklearn.manifold import SpectralEmbedding
import json, os

np.random.seed(0)
X = np.random.randn(20, 3)
se = SpectralEmbedding(n_components=2, n_neighbors=5, affinity='nearest_neighbors', random_state=0)
T = se.fit_transform(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/spectral_embedding.json', 'w') as f:
    json.dump({
        'X': X.tolist(),
        'expected': T.tolist()
    }, f)
