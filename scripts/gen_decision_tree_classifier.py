from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np, json, os

X, y = make_classification(n_samples=50, n_features=4, n_informative=3, n_redundant=0, random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
X_test, _ = make_classification(n_samples=10, n_features=4, n_informative=3, n_redundant=0, random_state=1)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/decision_tree_classifier.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
