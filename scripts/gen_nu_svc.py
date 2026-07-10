from sklearn.datasets import make_circles
from sklearn.svm import NuSVC
import json, os

X, y = make_circles(n_samples=160, noise=0.1, factor=0.4, random_state=0)
trainX, trainY = X[:120], y[:120]
testX, testY = X[120:], y[120:]

cases = {}
for nu in (0.3, 0.6):
    clf = NuSVC(nu=nu, kernel='rbf', gamma='scale')
    clf.fit(trainX, trainY)
    cases['nu_%s' % str(nu).replace('.', '_')] = {
        'nu': nu,
        'expected': clf.predict(testX).tolist(),
        'decision': clf.decision_function(testX).tolist(),
        'nSupport': clf.n_support_.tolist()
    }

os.makedirs('test_data', exist_ok=True)
with open('test_data/nu_svc.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'cases': cases
    }, f)
