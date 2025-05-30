import numpy as np
import json, os

np.random.seed(0)
X = np.random.randn(20, 3)
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
D2 = D ** 2
n = D.shape[0]
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J.dot(D2).dot(J)
vals, vecs = np.linalg.eigh(B)
idx = np.argsort(vals)[::-1][:2]
vals = vals[idx]
vecs = vecs[:, idx]
Y = vecs * np.sqrt(np.maximum(vals, 0))

os.makedirs('test_data', exist_ok=True)
with open('test_data/mds.json', 'w') as f:
    json.dump({'X': X.tolist(), 'expected': Y.tolist()}, f)
