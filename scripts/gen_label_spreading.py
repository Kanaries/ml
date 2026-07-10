from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np, json, os, warnings


def make_dataset(seed):
    """Two dense clusters + sparse bridge points + a small offset cluster.

    Node degrees differ wildly, so LabelPropagation (row-normalized
    affinity, hard clamping) and LabelSpreading (symmetric-normalized
    laplacian, soft clamping) genuinely disagree on this data, which makes
    the fixture discriminative between the two algorithms.
    """
    rng = np.random.RandomState(seed)
    dense_a = rng.randn(18, 2) * 0.3 + np.array([0.0, 0.0])
    dense_b = rng.randn(18, 2) * 0.3 + np.array([4.0, 0.0])
    bridge = np.stack([np.linspace(0.8, 3.2, 8), rng.randn(8) * 0.15], axis=1)
    offset = rng.randn(6, 2) * 0.4 + np.array([2.0, 2.5])
    X = np.vstack([dense_a, dense_b, bridge, offset])
    y = np.full(len(X), -1)
    y[0] = 0
    y[1] = 0
    y[18] = 1
    y[19] = 1
    y[44] = 2
    return X, y


def make_test(seed):
    rng = np.random.RandomState(seed + 1000)
    grid = np.array([
        [0.0, 0.2], [0.5, -0.3], [1.5, 0.1], [2.0, 0.0], [2.5, -0.1],
        [3.5, 0.2], [4.0, -0.2], [2.0, 2.3], [1.2, 1.2], [3.0, 1.5],
    ])
    return grid + rng.randn(*grid.shape) * 0.05


def find_discriminative_config():
    """Deterministic search for a (seed, gamma) on which LabelPropagation and
    LabelSpreading(alpha=0.2) disagree on >= 2 test predictions, so the
    fixture cannot pass with the two implementations swapped."""
    for seed in range(50):
        for gamma in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            X, y = make_dataset(seed)
            X_test = make_test(seed)
            lp = LabelPropagation(gamma=gamma)
            ls = LabelSpreading(gamma=gamma, alpha=0.2)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lp.fit(X, y)
                ls.fit(X, y)
            n_diff = int(np.sum(lp.predict(X_test) != ls.predict(X_test)))
            if n_diff >= 2:
                return seed, gamma
    raise AssertionError('no discriminative (seed, gamma) found')


seed, gamma = find_discriminative_config()
print(f'using seed={seed}, gamma={gamma}')

X, y = make_dataset(seed)
X_test = make_test(seed)
clf = LabelSpreading(gamma=gamma, alpha=0.2)
clf.fit(X, y)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

os.makedirs('test_data', exist_ok=True)
with open('test_data/label_spreading.json', 'w') as f:
    json.dump({
        'gamma': gamma,
        'trainX': X.tolist(),
        'trainY': y.tolist(),
        'testX': X_test.tolist(),
        'expected': pred.tolist(),
        'expectedProba': proba.tolist()
    }, f)
