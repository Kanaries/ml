from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
import json, os

X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=0)
reg = LinearSVR(random_state=0, max_iter=1000)
reg.fit(X, y)
X_test, y_test = make_regression(n_samples=10, n_features=4, noise=0.1, random_state=1)
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/linear_svr.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
