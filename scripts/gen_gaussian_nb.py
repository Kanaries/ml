from sklearn.naive_bayes import GaussianNB
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = np.vstack([
    rng.normal(loc=0.0, scale=1.0, size=(20, 3)),
    rng.normal(loc=3.0, scale=1.2, size=(20, 3)),
])
y = np.array([0] * 20 + [1] * 20)
X_test = np.vstack([
    rng.normal(loc=0.2, scale=1.0, size=(5, 3)),
    rng.normal(loc=2.8, scale=1.2, size=(5, 3)),
])
clf = GaussianNB()
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/gaussian_nb.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
