import numpy as np
from sklearn.manifold import TSNE
import json, os

np.random.seed(0)
X = np.random.randn(50, 4)
model = TSNE(n_components=2, perplexity=20, n_iter=250, learning_rate=200, init='random', random_state=0, method='exact')
Y = model.fit_transform(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/tsne.json', 'w') as f:
    json.dump({'X': X.tolist(), 'expected': Y.tolist()}, f)
