from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import json, os

X, y = make_regression(n_samples=80, n_features=4, n_informative=4, noise=0.4, random_state=0)
reg = ElasticNet(alpha=0.2, l1_ratio=0.4, max_iter=50000, tol=1e-12, random_state=0)
reg.fit(X, y)
X_test, _ = make_regression(n_samples=12, n_features=4, n_informative=4, noise=0.4, random_state=1)
pred = reg.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/elastic_net.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist()
    }, f)
