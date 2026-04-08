from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import json, os

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [5.0, 5.0],
    [5.0, 6.0],
    [6.0, 5.0],
])
y = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([
    [0.2, 0.3],
    [5.2, 5.1],
    [0.7, 0.1],
    [5.4, 5.8],
])
clf = RadiusNeighborsClassifier(radius=1.6)
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/radius_neighbors_classifier.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
