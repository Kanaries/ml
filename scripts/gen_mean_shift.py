import numpy as np
from sklearn.cluster import MeanShift
import json, os

np.random.seed(0)
X1 = np.random.randn(10,2) + np.array([0,0])
X2 = np.random.randn(10,2) + np.array([4,4])
X = np.vstack([X1, X2])
model = MeanShift(bandwidth=2)
labels = model.fit_predict(X)

os.makedirs('test_data', exist_ok=True)
with open('test_data/mean_shift.json', 'w') as f:
    json.dump({'X': X.tolist(), 'expected': labels.tolist()}, f)
