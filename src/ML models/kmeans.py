"""K-Means clustering of patients (unsupervised)."""
import importlib.util
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_LABELS
from src.ml_utils import _save

INFO = {
    "slug": "kmeans", "name": "K-Means Clustering",
    "type": "Unsupervised - Clustering", "family": "unsupervised",
    "summary": "Groups patients into k clusters with similar hospital-stay and utilisation profiles - no outcome labels used.",
    "vital_use": "Discovers patient segments (e.g. short routine stays vs. long, medication-heavy stays) and compares their readmission rates.",
    "how": [
        "K-Means picks k cluster centres, assigns every patient to the nearest centre, moves each centre to the mean of its patients and repeats until nothing changes. It only looks at the standardised numeric measurements (days in hospital, lab procedures, procedures, medications, prior visits, diagnoses, age) - the readmission label is NOT used for clustering.",
        "The number of clusters k is chosen with the elbow method (see the elbow chart): inertia always falls as k grows, and the 'elbow' is the point after which extra clusters give little improvement."],
    "why": [
        "Clustering shows whether natural groups of patients exist in the data. After clustering we check the readmission rate inside each cluster, which reveals whether a patient segment is more at risk - without the algorithm ever seeing the outcome.",
        "Cluster profiles (average days in hospital, medications, prior inpatient visits, age) give hospital staff an interpretable segmentation of their patient population."],
    "strengths": ["Fast and scalable", "Easy to interpret cluster centres"],
    "limits": ["Must choose k in advance", "Assumes round, similar-size clusters", "Sensitive to outliers"],
    "show_params": [],
}


def _elbow():
    spec = importlib.util.spec_from_file_location("km_elbow", Path(__file__).with_name("kmeans_elbow.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(data, elbow=None):
    mod = _elbow()
    elbow = elbow or mod.run(data)
    X, _ = mod.numeric_matrix(data)
    k = elbow["chosen_k"]
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_
    rng = np.random.RandomState(RANDOM_STATE)
    sidx = rng.choice(len(X), min(8000, len(X)), replace=False)
    sil = float(silhouette_score(X[sidx], labels[sidx]))
    dbi = float(davies_bouldin_score(X, labels))

    raw = data["X_train"].reset_index(drop=True)
    y = data["y_train"].to_numpy()
    prof_cols = [c for c in ["time_in_hospital", "num_lab_procedures", "num_medications",
                             "number_inpatient", "number_diagnoses", "age_num"] if c in raw]
    clusters = []
    for c in range(k):
        m = labels == c
        clusters.append({"cluster": c, "size": int(m.sum()), "pct": round(100 * m.mean(), 1),
                         "readmit_rate": round(100 * float(y[m].mean()), 1),
                         "means": {col: round(float(raw.loc[m, col].mean()), 1) for col in prof_cols}})

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    P = pca.fit_transform(X[sidx])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels[sidx], cmap="tab10", s=7, alpha=.65)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}% variance)")
    ax.set_title(f"K-Means clusters (k = {k}) projected with PCA", fontweight="bold")
    ax.legend(*sc.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.grid(alpha=.25)
    _save(fig, "cluster_plot.png")

    joblib.dump({"kmeans": km, "numeric_columns": data["num_cols"]}, MODELS_DIR / "kmeans.pkl", compress=3)
    return {"slug": INFO["slug"], "name": INFO["name"], "kind": "unsupervised", "k": k,
            "inertia": float(km.inertia_), "silhouette": sil, "davies_bouldin": dbi,
            "clusters": clusters, "profile_columns": prof_cols, "elbow": elbow,
            "n_samples": int(len(X)), "pca_variance": [float(v) for v in pca.explained_variance_ratio_],
            "params": {"n_clusters": str(k), "n_init": "10", "features": ", ".join(data["num_cols"])},
            "overall_readmit_rate": round(100 * float(y.mean()), 1)}
