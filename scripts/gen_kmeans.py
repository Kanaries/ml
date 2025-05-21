import numpy as np
from sklearn.cluster import KMeans
import json, os

np.random.seed(0)
X1 = np.random.randn(20,2) + np.array([0,0])
X2 = np.random.randn(20,2) + np.array([5,5])
X = np.vstack([X1, X2])
model = KMeans(n_clusters=2, n_init=10, random_state=0)
labels = model.fit_predict(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/kmeans.json', 'w') as f:
    json.dump({ 'X': X.tolist(), 'expected': labels.tolist() }, f)
