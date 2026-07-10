import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import json, os

# Clustered data (instead of pure noise) so the TS test can make structural
# assertions: t-SNE output is not comparable coordinate-wise (sign/rotation/
# local-optimum indeterminacy), but cluster separation must be preserved.
X, labels = make_blobs(n_samples=60, n_features=4, centers=3, cluster_std=0.5, random_state=0)
model = TSNE(n_components=2, perplexity=20, max_iter=250, learning_rate=200, init='random', random_state=0, method='exact')
Y = model.fit_transform(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/tsne.json', 'w') as f:
    json.dump({'X': X.tolist(), 'labels': labels.tolist(), 'expected': Y.tolist()}, f)
