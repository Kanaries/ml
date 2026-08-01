"""Generate the deterministic sklearn 1.9 oracle for the Phase 3 exit scenarios."""

import gzip
import json
import os

import numpy as np
import sklearn
from scipy.spatial.distance import pdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import (AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN,
                             KMeans, MeanShift, MiniBatchKMeans, SpectralClustering)
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import (fetch_20newsgroups, fetch_olivetti_faces, load_digits, load_iris,
                              make_blobs, make_circles, make_classification, make_moons, make_s_curve)
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import HuberRegressor, LogisticRegression, RANSACRegressor, TheilSenRegressor
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, trustworthiness
from sklearn.metrics import (accuracy_score, adjusted_rand_score, brier_score_loss, f1_score,
                             median_absolute_error)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

EXPECTED_SKLEARN = "1.9.0"
if sklearn.__version__ != EXPECTED_SKLEARN:
    raise RuntimeError(f"Phase 3 fixtures require sklearn {EXPECTED_SKLEARN}, got {sklearn.__version__}")


def values(array):
    return np.asarray(array).tolist()


def reconstruction_error(original, reconstructed):
    return float(np.linalg.norm(np.asarray(original) - np.asarray(reconstructed)))


fixture = {"schemaVersion": 1, "sklearnVersion": EXPECTED_SKLEARN}

# 1. Iris: exact frozen split and regularization search.
iris = load_iris()
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    iris.data, iris.target, test_size=.25, random_state=42, stratify=iris.target)
iris_search = GridSearchCV(Pipeline([
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
]), {"lr__C": [.1, 1., 10.]}, cv=5)
iris_search.fit(Xi_train, yi_train)
iris_pred = iris_search.predict(Xi_test)
fixture["iris"] = {
    "XTrain": values(Xi_train), "XTest": values(Xi_test),
    "yTrain": values(yi_train), "yTest": values(yi_test),
    "expected": {"accuracy": accuracy_score(yi_test, iris_pred), "predictions": values(iris_pred),
                 "bestC": iris_search.best_params_["lr__C"]},
}

# 2. Digits: full dataset, frozen stratified split.
digits = load_digits()
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    digits.data, digits.target, test_size=.2, random_state=42, stratify=digits.target)
digits_model = Pipeline([("scale", StandardScaler()), ("svc", SVC(C=10, gamma="scale"))])
digits_model.fit(Xd_train, yd_train)
digits_pred = digits_model.predict(Xd_test)
fixture["digits"] = {
    "XTrain": values(Xd_train), "XTest": values(Xd_test),
    "yTrain": values(yd_train), "yTest": values(yd_test),
    "expected": {"accuracy": accuracy_score(yd_test, digits_pred), "predictions": values(digits_pred)},
}

# 3. Text: official four-category train/test subsets, cached by sklearn.
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
text_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
text_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=42)
text_model = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
text_model.fit(text_train.data, text_train.target)
text_pred = text_model.predict(text_test.data)
fixture["text"] = {
    "trainDocuments": text_train.data, "testDocuments": text_test.data,
    "yTrain": values(text_train.target), "yTest": values(text_test.target),
    "expected": {"accuracy": accuracy_score(text_test.target, text_pred),
                 "macroF1": f1_score(text_test.target, text_pred, average="macro"),
                 "predictions": values(text_pred)},
}

# 4. Faces: deterministic 8x8 average pooling keeps the browser/CI covariance bounded.
faces = fetch_olivetti_faces(shuffle=False).images[:100]
pooled_faces = faces.reshape(100, 8, 8, 8, 8).mean(axis=(2, 4)).reshape(100, 64)
face_expected = {}
for name, model in {
    "pca": PCA(n_components=16, svd_solver="full"),
    "nmf": NMF(n_components=16, init="nndsvda", max_iter=1000, random_state=42),
    "fastICA": FastICA(n_components=16, whiten="unit-variance", max_iter=500, random_state=42, tol=1e-4),
}.items():
    transformed = model.fit_transform(pooled_faces)
    face_expected[name] = {
        "reconstructionError": reconstruction_error(pooled_faces, model.inverse_transform(transformed)),
        "shape": [16, 64],
    }
fixture["faces"] = {"X": values(pooled_faces), "expected": face_expected}

# 5. Anomaly comparison.
rng = np.random.RandomState(42)
inliers = np.r_[rng.normal(loc=(-2, -2), scale=.45, size=(90, 2)),
                rng.normal(loc=(2, 2), scale=.45, size=(90, 2))]
outliers = rng.uniform(low=-6, high=6, size=(20, 2))
Xa = np.r_[inliers, outliers]
ya = np.r_[np.ones(len(inliers), dtype=int), -np.ones(len(outliers), dtype=int)]
anomaly_models = {
    "isolationForest": IsolationForest(n_estimators=100, max_samples=128, contamination=.1, random_state=42),
    "oneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=.1),
    "localOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=.1),
    "ellipticEnvelope": EllipticEnvelope(contamination=.1, support_fraction=.8, random_state=42),
}
anomaly_expected = {}
for name, model in anomaly_models.items():
    pred = model.fit_predict(Xa)
    scores = model.negative_outlier_factor_ if name == "localOutlierFactor" else model.decision_function(Xa)
    anomaly_expected[name] = {"labels": values(pred), "scores": values(scores),
                              "outlierRecall": float(np.mean(pred[ya == -1] == -1))}
fixture["anomaly"] = {"X": values(Xa), "truth": values(ya), "expected": anomaly_expected}

# 6. Three varied shapes; 120 samples per shape is the frozen browser/CI projection.
cluster_sets = {}
X, y = make_blobs(n_samples=120, centers=3, cluster_std=.55, random_state=42)
cluster_sets["blobs"] = (X, y, 3, .35, .9)
X, y = make_moons(n_samples=120, noise=.06, random_state=42)
cluster_sets["moons"] = (X, y, 2, .28, .7)
X, y = make_circles(n_samples=120, factor=.45, noise=.04, random_state=42)
cluster_sets["circles"] = (X, y, 2, .22, .65)
fixture["clustering"] = {"datasets": {}}
for dataset_name, (X, y, k, eps, bandwidth) in cluster_sets.items():
    Xs = StandardScaler().fit_transform(X)
    models = {
        "kMeans": KMeans(n_clusters=k, n_init=10, random_state=42),
        "miniBatchKMeans": MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=32, random_state=42),
        "dbscan": DBSCAN(eps=eps, min_samples=5),
        "agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "meanShift": MeanShift(bandwidth=bandwidth),
        "spectral": SpectralClustering(n_clusters=k, affinity="rbf", gamma=1, n_init=10, random_state=42),
        "birch": Birch(n_clusters=k, threshold=.35),
        "affinityPropagation": AffinityPropagation(damping=.7, random_state=42),
        "gaussianMixture": GaussianMixture(n_components=k, n_init=1, random_state=42),
    }
    expected = {}
    for name, model in models.items():
        pred = model.fit_predict(Xs)
        expected[name] = {"labels": values(pred), "ari": adjusted_rand_score(y, pred)}
    fixture["clustering"]["datasets"][dataset_name] = {
        "X": values(X), "truth": values(y), "nClusters": k, "eps": eps, "bandwidth": bandwidth,
        "expected": expected,
    }

# 7. S-curve browser/CI projection. Metrics are invariant to sign/rotation.
Xm, _ = make_s_curve(n_samples=120, noise=.05, random_state=42)
manifold_models = {
    "tsne": TSNE(n_components=2, perplexity=20, learning_rate=200, max_iter=500, init="random", random_state=42),
    "lle": LocallyLinearEmbedding(n_neighbors=12, n_components=2, eigen_solver="dense"),
    "mds": MDS(n_components=2, random_state=42, n_init=1, max_iter=300, normalized_stress="auto", init="random"),
    "isomap": Isomap(n_neighbors=12, n_components=2, eigen_solver="dense"),
    "spectralEmbedding": SpectralEmbedding(n_components=2, n_neighbors=12, random_state=42, eigen_solver="lobpcg"),
}
manifold_expected = {}
original_distances = pdist(Xm)
for name, model in manifold_models.items():
    embedding = model.fit_transform(Xm)
    manifold_expected[name] = {
        "trustworthiness": trustworthiness(Xm, embedding, n_neighbors=10),
        "distanceCorrelation": float(np.corrcoef(original_distances, pdist(embedding))[0, 1]),
    }
fixture["manifold"] = {"X": values(Xm), "expected": manifold_expected}

# 8. Binary feature-selection workflow (selectors use coefficient importance).
Xf, yf = make_classification(n_samples=200, n_features=12, n_informative=4, n_redundant=0,
                             n_repeated=0, random_state=42, shuffle=False)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    Xf, yf, test_size=.25, random_state=42, stratify=yf)
feature_expected = {}
for name, selector in {
    "selectFromModel": SelectFromModel(LogisticRegression(max_iter=2000, solver="lbfgs"), max_features=4, threshold=-np.inf),
    "rfe": RFE(LogisticRegression(max_iter=2000, solver="lbfgs"), n_features_to_select=4, step=1),
}.items():
    pipe = Pipeline([("scale", StandardScaler()), ("selector", selector),
                     ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    pipe.fit(Xf_train, yf_train)
    pred = pipe.predict(Xf_test)
    feature_expected[name] = {"support": values(pipe.named_steps["selector"].get_support()),
                              "accuracy": accuracy_score(yf_test, pred)}
fixture["featureSelection"] = {
    "XTrain": values(Xf_train), "XTest": values(Xf_test),
    "yTrain": values(yf_train), "yTest": values(yf_test), "expected": feature_expected,
}

# 9. Fixed held-out calibration evaluation with internal deterministic 3-fold CV.
Xc, yc = make_classification(n_samples=600, n_features=10, n_informative=5, n_redundant=2,
                             class_sep=.8, flip_y=.08, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=.25, random_state=42, stratify=yc)
calibration_expected = {}
for method in ["sigmoid", "isotonic"]:
    model = CalibratedClassifierCV(estimator=GaussianNB(), method=method, cv=3, ensemble=True)
    model.fit(Xc_train, yc_train)
    probabilities = model.predict_proba(Xc_test)[:, 1]
    calibration_expected[method] = {"probabilities": values(probabilities),
                                    "brier": brier_score_loss(yc_test, probabilities),
                                    "accuracy": accuracy_score(yc_test, model.predict(Xc_test))}
fixture["calibration"] = {
    "XTrain": values(Xc_train), "XTest": values(Xc_test),
    "yTrain": values(yc_train), "yTest": values(yc_test), "expected": calibration_expected,
}

# 10. Robust regression with 20 frozen vertical outliers.
rng = np.random.RandomState(42)
Xr = rng.uniform(-4, 4, size=(200, 1))
yr_clean = 1.5 + 2.75 * Xr[:, 0] + rng.normal(0, .25, size=200)
yr = yr_clean.copy()
outlier_indices = rng.choice(200, 20, replace=False)
yr[outlier_indices] += rng.choice([-1, 1], size=20) * rng.uniform(12, 20, size=20)
robust_expected = {}
for name, model in {
    "huber": HuberRegressor(epsilon=1.35, max_iter=500, tol=1e-8),
    "ransac": RANSACRegressor(random_state=42, residual_threshold=.75, max_trials=100),
    "theilSen": TheilSenRegressor(random_state=42, max_subpopulation=1000, max_iter=500),
}.items():
    model.fit(Xr, yr)
    pred = model.predict(Xr)
    fitted = model.estimator_ if name == "ransac" else model
    robust_expected[name] = {"coef": values(fitted.coef_), "intercept": float(fitted.intercept_),
                             "medianAbsoluteError": median_absolute_error(yr_clean, pred)}
    if name == "ransac":
        robust_expected[name]["inlierMask"] = values(model.inlier_mask_)
fixture["robustRegression"] = {"X": values(Xr), "y": values(yr), "yClean": values(yr_clean),
                               "outlierIndices": values(outlier_indices), "expected": robust_expected}

payload = json.dumps(fixture, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
target = os.path.join(os.path.dirname(__file__), "..", "test_data", "phase3_e2e.json.gz")
with open(target, "wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
        zipped.write(payload)
print(f"wrote {target}: {len(payload)} bytes JSON")
