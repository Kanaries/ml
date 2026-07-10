from sklearn.datasets import make_circles, make_blobs, make_classification
from sklearn.svm import SVC, LinearSVC
import json, os

# --- rbf binary: concentric circles (non-linear boundary) ---
X, y = make_circles(n_samples=160, noise=0.1, factor=0.4, random_state=0)
rbf_trainX, rbf_trainY = X[:120], y[:120]
rbf_testX, rbf_testY = X[120:], y[120:]

rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf.fit(rbf_trainX, rbf_trainY)
rbf_pred = rbf.predict(rbf_testX)
rbf_decision = rbf.decision_function(rbf_testX)

# --- linear binary: keeps LinearSVC / SVC(kernel='linear') coverage ---
lin_trainX, lin_trainY = make_classification(
    n_samples=50, n_features=2, n_informative=2, n_redundant=0,
    n_classes=2, class_sep=2.0, random_state=0)
lin_testX, _ = make_classification(
    n_samples=10, n_features=2, n_informative=2, n_redundant=0,
    n_classes=2, class_sep=2.0, random_state=1)

lin = SVC(kernel='linear', C=1.0)
lin.fit(lin_trainX, lin_trainY)
lin_pred = lin.predict(lin_testX)
lin_decision = lin.decision_function(lin_testX)

linsvc = LinearSVC(max_iter=1000, dual=True)
linsvc.fit(lin_trainX, lin_trainY)
lin_pred_linsvc = linsvc.predict(lin_testX)

# --- 3-class blobs: one-vs-one multiclass ---
Xm, ym = make_blobs(n_samples=120, centers=3, cluster_std=1.5, random_state=0)
multi_trainX, multi_trainY = Xm[:90], ym[:90]
multi_testX = Xm[90:]

multi = SVC(kernel='rbf', C=1.0, gamma='scale')
multi.fit(multi_trainX, multi_trainY)
multi_pred = multi.predict(multi_testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/svc.json', 'w') as f:
    json.dump({
        'rbf': {
            'trainX': rbf_trainX.tolist(),
            'trainY': rbf_trainY.tolist(),
            'testX': rbf_testX.tolist(),
            'expected': rbf_pred.tolist(),
            'decision': rbf_decision.tolist(),
            'nSupport': rbf.n_support_.tolist()
        },
        'linear': {
            'trainX': lin_trainX.tolist(),
            'trainY': lin_trainY.tolist(),
            'testX': lin_testX.tolist(),
            'expected': lin_pred.tolist(),
            'decision': lin_decision.tolist(),
            'nSupport': lin.n_support_.tolist(),
            'expected_linsvc': lin_pred_linsvc.tolist()
        },
        'multiclass': {
            'trainX': multi_trainX.tolist(),
            'trainY': multi_trainY.tolist(),
            'testX': multi_testX.tolist(),
            'expected': multi_pred.tolist(),
            'nSupport': multi.n_support_.tolist()
        }
    }, f)
