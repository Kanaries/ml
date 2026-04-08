from sklearn.neighbors import NearestCentroid
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = np.vstack([
    rng.normal(loc=[0.0, 0.0], scale=0.4, size=(20, 2)),
    rng.normal(loc=[3.0, 3.0], scale=0.4, size=(20, 2)),
])
y = np.array([0] * 20 + [1] * 20)
X_test = np.array([
    [0.2, 0.1],
    [2.8, 3.2],
    [0.0, -0.1],
    [3.4, 2.7],
])
clf = NearestCentroid()
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/nearest_centroid.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
