from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np, json, os

X, y = make_classification(n_samples=60, n_features=5, n_informative=3, n_classes=3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
X_test, _ = make_classification(n_samples=10, n_features=5, n_informative=3, n_classes=3, random_state=1)
pred = knn.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/knn.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
