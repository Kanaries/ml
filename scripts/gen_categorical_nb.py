from sklearn.naive_bayes import CategoricalNB
import numpy as np
import json, os

# Deliberately IMBALANCED classes (36/12) with class-conditional categories:
# the class prior must not be alpha-smoothed (see gen_bernoulli_nb.py). The
# test-point indices were selected so that alpha-smoothing the prior flips
# several predictions.
rng = np.random.RandomState(18)
X = np.vstack([
    rng.choice(3, size=(36, 5), p=[0.6, 0.3, 0.1]),
    rng.choice(3, size=(12, 5), p=[0.1, 0.3, 0.6]),
]).astype(int)
y = np.array([0] * 36 + [1] * 12)
T = rng.choice(3, size=(80, 5)).astype(int)
keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 59]
X_test = T[keep]

clf = CategoricalNB()
clf.fit(X, y)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/categorical_nb.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
