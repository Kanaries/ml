from sklearn.naive_bayes import CategoricalNB
import numpy as np
import json, os

rng = np.random.RandomState(0)
X = rng.randint(3, size=(50, 5))
y = rng.randint(2, size=50)
X_test = rng.randint(3, size=(10, 5))
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
