from sklearn.datasets import make_classification
from sklearn.semi_supervised import LabelPropagation
import numpy as np, json, os

X, y = make_classification(n_samples=60, n_features=5, n_informative=3,
                           n_classes=3, random_state=0)
# randomly unlabel 40% of data
rng = np.random.RandomState(0)
mask = rng.rand(len(y)) < 0.4
trainY = y.copy()
trainY[mask] = -1
clf = LabelPropagation()
clf.fit(X, trainY)
X_test, _ = make_classification(n_samples=10, n_features=5, n_informative=3,
                                n_classes=3, random_state=1)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/label_propagation.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': trainY.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
