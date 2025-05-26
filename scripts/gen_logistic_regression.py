from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import json, os

X, y = make_classification(n_samples=60, n_features=5, n_informative=3,
                           n_classes=2, random_state=0)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
X_test, _ = make_classification(n_samples=10, n_features=5, n_informative=3,
                                n_classes=2, random_state=1)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/logistic_regression.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
