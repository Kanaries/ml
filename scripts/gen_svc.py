from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC, NuSVC
import json, os

X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0,
                           n_classes=2, class_sep=2.0, random_state=0)
X_test, _ = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0,
                                n_classes=2, class_sep=2.0, random_state=1)

svc = SVC(kernel='linear', max_iter=1000)
svc.fit(X, y)
pred_svc = svc.predict(X_test)

linsvc = LinearSVC(max_iter=1000, dual=True)
linsvc.fit(X, y)
pred_linsvc = linsvc.predict(X_test)

nusvc = NuSVC(kernel='linear', max_iter=1000)
nusvc.fit(X, y)
pred_nusvc = nusvc.predict(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/svc.json', 'w') as f:
    json.dump({
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected_svc': pred_svc.tolist(),
        'expected_linsvc': pred_linsvc.tolist(),
        'expected_nusvc': pred_nusvc.tolist()
    }, f)
