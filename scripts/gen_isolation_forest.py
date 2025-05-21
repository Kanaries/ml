import numpy as np
from sklearn.ensemble import IsolationForest
import json, os

np.random.seed(0)
X = np.random.randn(40, 2)
clf = IsolationForest(random_state=0, contamination=0.1)
clf.fit(X)
X_test = np.random.randn(10, 2)
pred = clf.predict(X_test)
pred = [0 if p == 1 else 1 for p in pred]

os.makedirs('test_data', exist_ok=True)
with open('test_data/isolation_forest.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'testX': X_test.tolist(),
        'expected': pred
    }, f)
