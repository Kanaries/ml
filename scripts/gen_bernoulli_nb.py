from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
import os

rng = np.random.RandomState(0)
X = rng.randint(2, size=(30, 5))
y = rng.randint(2, size=30)
X_test = rng.randint(2, size=(10, 5))
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
