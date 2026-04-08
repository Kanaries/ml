from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier
import json, os

X, y = make_classification(
    n_samples=90,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=0,
)
clf = RidgeClassifier(alpha=1.0)
clf.fit(X, y)
X_test, _ = make_classification(
    n_samples=15,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=1,
)
pred = clf.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/ridge_classifier.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
