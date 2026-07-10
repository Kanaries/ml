from sklearn.naive_bayes import ComplementNB
import numpy as np
import json, os

# Deliberately IMBALANCED classes (30/10): ComplementNB's JLL must NOT include
# the class prior for multi-class problems (sklearn only adds it for a single
# class). Balanced fixtures cancel the prior in the argmax and structurally
# mask that bug. The test-point indices below were selected so that adding the
# prior to the JLL flips several predictions — i.e. the fixture discriminates
# the correct formula from the buggy one.
rng = np.random.RandomState(0)
X = np.vstack([
    rng.poisson(lam=[3, 1, 0.5], size=(30, 3)),
    rng.poisson(lam=[0.5, 2.5, 3], size=(10, 3)),
]).astype(int)
y = np.array([0] * 30 + [1] * 10)
T = rng.poisson(lam=[1.5, 1.5, 1.5], size=(60, 3)).astype(int)
keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 18, 23, 27, 28, 41]
X_test = T[keep]

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
