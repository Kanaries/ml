from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
import os

# Deliberately IMBALANCED classes (24/8) with class-conditional features:
# sklearn never smooths the class prior with alpha, so an alpha-smoothed prior
# shifts the decision boundary on imbalanced data. The test-point indices were
# selected so that alpha-smoothing the prior flips several predictions.
rng = np.random.RandomState(4)
p0 = np.array([0.8, 0.6, 0.3, 0.2, 0.5])
p1 = np.array([0.2, 0.4, 0.7, 0.8, 0.5])
X = np.vstack([
    (rng.rand(24, 5) < p0).astype(int),
    (rng.rand(8, 5) < p1).astype(int),
])
y = np.array([0] * 24 + [1] * 8)
T = (rng.rand(80, 5) < 0.5).astype(int)
keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 60]
X_test = T[keep]

clf = BernoulliNB()
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/bernoulli_nb.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
