from sklearn.naive_bayes import ComplementNB
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = np.zeros((40, 3), dtype=int)
y = np.array([0] * 20 + [1] * 20)
X[:20] = rng.poisson(lam=[3, 1, 0.5], size=(20, 3))
X[20:] = rng.poisson(lam=[0.5, 2.5, 3], size=(20, 3))
X_test = np.array([
    [4, 1, 0],
    [0, 3, 3],
    [2, 1, 1],
    [1, 2, 2],
], dtype=int)
clf = ComplementNB()
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/complement_nb.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
