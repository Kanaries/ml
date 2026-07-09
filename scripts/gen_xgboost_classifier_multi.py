import json, os, sys

try:
    from xgboost import XGBClassifier
except ImportError:
    print('WARNING: xgboost not installed, skipping xgboost_classifier_multi fixture '
          '(the committed fixture stays in place)', file=sys.stderr)
    sys.exit(0)

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=180,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    random_state=0,
)
trainX = X[:130]
trainY = y[:130]
testX = X[130:]

clf = XGBClassifier(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=6,
    reg_lambda=1,
    tree_method='exact',
    objective='multi:softprob',
    random_state=0,
)
clf.fit(trainX, trainY)
pred = clf.predict(testX)

os.makedirs('test_data', exist_ok=True)
with open('test_data/xgboost_classifier_multi.json', 'w') as f:
    json.dump({
        'trainX': trainX.tolist(),
        'trainY': trainY.tolist(),
        'testX': testX.tolist(),
        'expected': [int(v) for v in pred]
    }, f)
