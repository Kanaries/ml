import numpy as np
from sklearn.cluster import DBSCAN
import json, os

np.random.seed(0)
X1 = np.random.randn(10,2)
X2 = np.random.randn(10,2) + np.array([5,5])
X = np.vstack([X1,X2])
model = DBSCAN(eps=0.8, min_samples=3)
labels = model.fit_predict(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/dbscan.json', 'w') as f:
    json.dump({'X': X.tolist(), 'expected': labels.tolist()}, f)
